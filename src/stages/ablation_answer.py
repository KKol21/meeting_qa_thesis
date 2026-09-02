"""Stage 3: run one answer model over oracle or retrieved evidence."""

import argparse
import json
from pathlib import Path
from statistics import mean

from rouge_score import rouge_scorer

from meeting_qa_chunking.answering import (
    ANSWER_INSTRUCTION,
    ROUGE_TYPES,
    build_answer_prompt,
    score_answer,
)
from meeting_qa_chunking.artifacts import read_retrieval
from meeting_qa_chunking.experiment import (
    EXPERIMENT_VERSION,
    select_meeting_paths,
    write_json,
)
from meeting_qa_chunking.local_model import LocalChatModel
from meeting_qa_chunking.pipeline import (
    prepare_oracle_evidence,
    prepare_retrieved_evidence,
)
from meeting_qa_chunking.qmsum import load_meeting


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")
DEFAULT_LUMBER_DIR = Path("runs/lumber/qmsum")
DEFAULT_RETRIEVAL_DIR = Path("runs/ablations/full/retrieval")


def summarize(output_dir: Path, meeting_ids: list[str]) -> dict[str, object]:
    results = [
        json.loads((output_dir / f"{meeting_id}.json").read_text(encoding="utf-8"))
        for meeting_id in meeting_ids
    ]
    condition_names = list(results[0]["conditions"])
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": "answers",
        "source": results[0]["source"],
        "answer_model": results[0]["answer_model"],
        "meeting_ids": meeting_ids,
        "meeting_count": len(meeting_ids),
        "question_count": sum(result["question_count"] for result in results),
        "conditions": results[0]["conditions"],
        "macro_average_rouge_f1": {
            condition: {
                rouge_type: mean(
                    question["results"][condition]["rouge_f1"][rouge_type]
                    for result in results
                    for question in result["questions"]
                )
                for rouge_type in ROUGE_TYPES
            }
            for condition in condition_names
        },
        "model_calls": sum(result["model_calls"] for result in results),
        "cache_hits": sum(result["cache_hits"] for result in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--lumber-dir", type=Path, default=DEFAULT_LUMBER_DIR)
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", choices=("oracle", "retrieval"), required=True)
    parser.add_argument("--condition", action="append", dest="conditions")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meetings", nargs="+")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--prequantized", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.source == "oracle" and args.conditions:
        raise ValueError("--condition only applies to retrieved evidence")
    paths = select_meeting_paths(
        args.data_dir, args.count, args.seed, args.meetings
    )
    expected_model = {
        "tag": args.model_tag,
        "model": args.model,
        "revision": args.revision,
        "prequantized": args.prequantized,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "instruction": ANSWER_INSTRUCTION,
    }
    # Prompt, model, and evidence conditions all participate in resume checks.
    pending = []
    for path in paths:
        output_path = args.output_dir / path.name
        if args.source == "oracle":
            expected_conditions = {"oracle": {"source": "annotated evidence"}}
        else:
            retrieval_path = args.retrieval_dir / path.name
            if not retrieval_path.exists():
                raise FileNotFoundError(retrieval_path)
            retrieval = read_retrieval(retrieval_path)
            names = args.conditions or list(retrieval["configurations"])
            expected_conditions = {
                name: retrieval["configurations"][name] for name in names
            }
        if output_path.exists():
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                saved.get("experiment_version") == EXPERIMENT_VERSION
                and saved.get("source") == args.source
                and saved.get("answer_model") == expected_model
                and saved.get("conditions") == expected_conditions
            ):
                print(f"Answers {path.stem}: existing", flush=True)
                continue
        pending.append((path, output_path))

    # Avoid allocating a GPU model when every requested meeting is complete.
    model = None
    if pending:
        model = LocalChatModel(
            model_name=args.model,
            revision=args.revision,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            cache_dir=Path(".cache/answers"),
            prequantized=args.prequantized,
        )
        print(f"Answer model device: {model.device}", flush=True)

    for path, output_path in pending:
        meeting = load_meeting(path)
        if args.source == "oracle":
            conditions = {"oracle": {"source": "annotated evidence"}}
            oracle = prepare_oracle_evidence(meeting)
            prepared = [{"oracle": evidence} for evidence in oracle]
        else:
            conditions, prepared = prepare_retrieved_evidence(
                meeting,
                args.retrieval_dir / path.name,
                args.lumber_dir,
                args.conditions,
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
            "meeting_id": meeting.id,
            "question_count": len(questions),
            "source": args.source,
            "conditions": conditions,
            "answer_model": expected_model,
            "model_calls": model.model_calls - calls_before,
            "cache_hits": model.cache_hits - hits_before,
            "questions": questions,
        }
        write_json(output_path, result)
        print(f"Answers {meeting.id}: saved", flush=True)

    meeting_ids = [path.stem for path in paths]
    summary = summarize(args.output_dir, meeting_ids)
    write_json(args.output_dir / "summary.json", summary)
    print(f"Answer summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
