"""Stage 3: answer one preset-defined oracle or retrieval condition set."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean

from meeting_qa_chunking.answering import (
    ANSWER_INSTRUCTION,
    ROUGE_TYPES,
    build_answer_prompt,
    score_answer,
)
from meeting_qa_chunking.artifacts import (
    EXPERIMENT_VERSION,
    make_provenance,
    questions_complete,
    read_answers,
    read_retrieval,
    sha256_file,
    write_json,
)
from meeting_qa_chunking.config import load_run_config
from meeting_qa_chunking.evidence_preparation import (
    prepare_oracle_evidence,
    prepare_retrieved_evidence,
)
from meeting_qa_chunking.qmsum import load_meeting
from meeting_qa_chunking.selection import select_meeting_paths


def summarize(output_dir: Path, meeting_ids: list[str]) -> dict[str, object]:
    results = [
        json.loads((output_dir / f"{meeting_id}.json").read_text(encoding="utf-8"))
        for meeting_id in meeting_ids
    ]
    conditions = list(results[0]["conditions"])
    per_meeting = {
        result["meeting_id"]: {
            condition: {
                rouge_type: mean(
                    question["results"][condition]["rouge_f1"][rouge_type]
                    for question in result["questions"]
                )
                for rouge_type in ROUGE_TYPES
            }
            for condition in conditions
        }
        for result in results
    }
    paired = {}
    if results[0]["source"] == "retrieval":
        for name, condition in results[0]["conditions"].items():
            if condition["chunker"] != "lumber":
                continue
            suffix = f"{condition['retriever']}__w{condition['evidence_words']}"
            for baseline in ("turn_packed", "word_packed"):
                baseline_name = f"{baseline}__{suffix}"
                if baseline_name not in conditions:
                    continue
                comparison = f"lumber_minus_{baseline}__{suffix}"
                paired[comparison] = {
                    rouge_type: mean(
                        per_meeting[meeting_id][name][rouge_type]
                        - per_meeting[meeting_id][baseline_name][rouge_type]
                        for meeting_id in meeting_ids
                    )
                    for rouge_type in ROUGE_TYPES
                }
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": "answers",
        "source": results[0]["source"],
        "answer_model": results[0]["answer_model"],
        "meeting_ids": meeting_ids,
        "meeting_count": len(meeting_ids),
        "question_count": sum(result["question_count"] for result in results),
        "conditions": results[0]["conditions"],
        "question_average_rouge_f1": {
            condition: {
                rouge_type: mean(
                    question["results"][condition]["rouge_f1"][rouge_type]
                    for result in results
                    for question in result["questions"]
                )
                for rouge_type in ROUGE_TYPES
            }
            for condition in conditions
        },
        "per_meeting": per_meeting,
        "meeting_average_rouge_f1": {
            condition: {
                rouge_type: mean(
                    per_meeting[meeting_id][condition][rouge_type]
                    for meeting_id in meeting_ids
                )
                for rouge_type in ROUGE_TYPES
            }
            for condition in conditions
        },
        "paired_meeting_average": paired,
        "model_calls": sum(result["model_calls"] for result in results),
        "cache_hits": sum(result["cache_hits"] for result in results),
        "artifact_hashes": {
            meeting_id: sha256_file(output_dir / f"{meeting_id}.json")
            for meeting_id in meeting_ids
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--answer-stage", required=True)
    args = parser.parse_args()

    from rouge_score import rouge_scorer
    from meeting_qa_chunking.local_model import LocalChatModel

    run = load_run_config(args.preset)
    stage = run.answer_stage(args.answer_stage)
    generation = run.generation
    meeting_ids = run.meeting_ids()
    paths = select_meeting_paths(run.data_dir, len(meeting_ids), 0, meeting_ids)
    output_dir = run.answers_dir / stage.name
    answer_model = {
        **asdict(stage.model),
        "max_new_tokens": generation.max_new_tokens,
        "temperature": generation.temperature,
        "seed": generation.seed,
    }

    pending = []
    for path in paths:
        meeting = load_meeting(path)
        inputs = {"meeting": path}
        if stage.source == "oracle":
            conditions = {"oracle": {"source": "annotated evidence"}}
        else:
            retrieval_path = run.retrieval_dir / path.name
            lumber_path = run.lumber_dir / path.name
            if not retrieval_path.exists():
                raise FileNotFoundError(retrieval_path)
            retrieval = read_retrieval(retrieval_path)
            conditions = retrieval["configurations"]
            inputs["retrieval"] = retrieval_path
            if any(item["chunker"] == "lumber" for item in conditions.values()):
                if not lumber_path.exists():
                    raise FileNotFoundError(lumber_path)
                inputs["segmentation"] = lumber_path

        effective_config = {
            "experiment_version": EXPERIMENT_VERSION,
            "source": stage.source,
            "conditions": conditions,
            "model": answer_model,
            "prompt": ANSWER_INSTRUCTION,
            "rouge_types": ROUGE_TYPES,
            "rouge_stemmer": True,
        }
        provenance = make_provenance(
            "answers", effective_config, inputs, args.preset
        )
        output_path = output_dir / path.name
        if output_path.exists():
            try:
                saved = read_answers(output_path)
            except ValueError:
                saved = {}
            if (
                saved.get("provenance", {}).get("fingerprint")
                == provenance["fingerprint"]
                and saved.get("source") == stage.source
                and saved.get("conditions") == conditions
                and saved.get("answer_model") == answer_model
                and questions_complete(
                    saved,
                    meeting.id,
                    [question.text for question in meeting.questions],
                    set(conditions),
                )
            ):
                print(f"Answers {path.stem}: existing", flush=True)
                continue
        pending.append((meeting, output_path, provenance, conditions))

    model = None
    if pending:
        model = LocalChatModel(
            model_name=stage.model.name,
            revision=stage.model.revision,
            max_new_tokens=generation.max_new_tokens,
            seed=generation.seed,
            temperature=generation.temperature,
            cache_dir=Path(".cache/answers"),
            prequantized=stage.model.prequantized,
        )
        print(f"Answer model device: {model.device}", flush=True)

    for meeting, output_path, provenance, conditions in pending:
        if stage.source == "oracle":
            prepared = [
                {"oracle": evidence}
                for evidence in prepare_oracle_evidence(meeting)
            ]
        else:
            conditions, prepared = prepare_retrieved_evidence(
                meeting,
                run.retrieval_dir / f"{meeting.id}.json",
                run.lumber_dir,
                requested_conditions=None,
            )

        calls_before = model.model_calls
        hits_before = model.cache_hits
        scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
        questions = []
        for question_index, (question, evidence_by_condition) in enumerate(
            zip(meeting.questions, prepared)
        ):
            answers = {}
            for condition, evidence in evidence_by_condition.items():
                answer = model(
                    build_answer_prompt(question.text, evidence["text"])
                )
                answers[condition] = {
                    "answer": answer,
                    "rouge_f1": score_answer(
                        scorer, question.reference_answer, answer
                    ),
                    "cache_hit": model.last_cache_hit,
                    **evidence["metadata"],
                }
                source = "cache" if model.last_cache_hit else "model"
                print(
                    f"{meeting.id} question {question_index} "
                    f"{condition} ({source})",
                    flush=True,
                )
            questions.append(
                {
                    "question_index": question_index,
                    "question": question.text,
                    "reference_answer": question.reference_answer,
                    "results": answers,
                }
            )

        result = {
            "experiment_version": EXPERIMENT_VERSION,
            "provenance": provenance,
            "meeting_id": meeting.id,
            "question_count": len(questions),
            "source": stage.source,
            "conditions": conditions,
            "answer_model": answer_model,
            "model_calls": model.model_calls - calls_before,
            "cache_hits": model.cache_hits - hits_before,
            "questions": questions,
        }
        write_json(output_path, result)
        print(f"Answers {meeting.id}: saved", flush=True)

    write_json(output_dir / "summary.json", summarize(output_dir, meeting_ids))
    print(f"Answer summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
