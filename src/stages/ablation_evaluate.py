"""Stage 4: evaluate preset answers with BERTScore and an LLM judge."""

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
from pathlib import Path
from statistics import mean

from meeting_qa_chunking.artifacts import (
    EXPERIMENT_VERSION,
    make_provenance,
    read_answers,
    read_answer_summary,
    write_json,
)
from meeting_qa_chunking.config import load_run_config
from meeting_qa_chunking.evidence import render_gold_evidence
from meeting_qa_chunking.judging import (
    JUDGE_INSTRUCTION,
    build_judge_prompt,
    parse_judgment,
)
from meeting_qa_chunking.qmsum import load_meeting


BERTSCORE_PACKAGE_VERSION = "0.3.13"


def evaluation_config(run) -> dict[str, object]:
    spec = run.evaluation
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "bertscore": {
            "package_version": BERTSCORE_PACKAGE_VERSION,
            "model": asdict(spec.bertscore_model),
            "layers": spec.bertscore_layers,
            "batch_size": spec.bertscore_batch_size,
            "rescale_with_baseline": False,
        },
        "judge": {
            "model": asdict(spec.judge_model),
            "max_new_tokens": spec.judge_max_new_tokens,
            "temperature": spec.judge_temperature,
            "seed": spec.judge_seed,
            "prompt": JUDGE_INSTRUCTION,
        },
    }


def load_stage(answer_dir: Path, data_dir: Path):
    answer_summary = read_answer_summary(answer_dir / "summary.json")
    meetings = {
        meeting_id: load_meeting(data_dir / f"{meeting_id}.json")
        for meeting_id in answer_summary["meeting_ids"]
    }
    records = []
    for meeting_id, meeting in meetings.items():
        saved = read_answers(answer_dir / f"{meeting_id}.json")
        for question_result in saved["questions"]:
            question_index = question_result["question_index"]
            question = meeting.questions[question_index]
            gold_evidence, _turn_ids = render_gold_evidence(question, meeting)
            for condition, result in question_result["results"].items():
                records.append(
                    {
                        "meeting_id": meeting_id,
                        "question_index": question_index,
                        "condition": condition,
                        "question": question.text,
                        "reference_answer": question.reference_answer,
                        "gold_evidence": gold_evidence,
                        "candidate_answer": result["answer"],
                    }
                )
    return answer_summary, records


def add_bertscore(
    scorer,
    records: list[dict[str, object]],
    batch_size: int,
) -> None:
    precision, recall, f1 = scorer.score(
        [record["candidate_answer"] for record in records],
        [record["reference_answer"] for record in records],
        batch_size=batch_size,
    )
    for record, p, r, f in zip(
        records, precision.tolist(), recall.tolist(), f1.tolist()
    ):
        record["bertscore"] = {"precision": p, "recall": r, "f1": f}


def _metrics(records: list[dict[str, object]]) -> dict[str, object]:
    distribution = Counter(record["judge"]["score"] for record in records)
    return {
        "answer_count": len(records),
        "bertscore": {
            metric: mean(record["bertscore"][metric] for record in records)
            for metric in ("precision", "recall", "f1")
        },
        "judge_mean": mean(record["judge"]["score"] for record in records),
        "judge_distribution": {
            str(score): distribution[score] for score in (1, 2, 3)
        },
    }


def summarize_stage(result: dict[str, object]) -> dict[str, object]:
    records = result["records"]
    meeting_ids = result["answer_summary"]["meeting_ids"]
    conditions = list(result["answer_summary"]["conditions"])
    per_meeting = {
        meeting_id: {
            condition: _metrics(
                [
                    record
                    for record in records
                    if record["meeting_id"] == meeting_id
                    and record["condition"] == condition
                ]
            )
            for condition in conditions
        }
        for meeting_id in meeting_ids
    }
    meeting_average = {
        condition: {
            "bertscore": {
                metric: mean(
                    per_meeting[meeting_id][condition]["bertscore"][metric]
                    for meeting_id in meeting_ids
                )
                for metric in ("precision", "recall", "f1")
            },
            "judge_mean": mean(
                per_meeting[meeting_id][condition]["judge_mean"]
                for meeting_id in meeting_ids
            ),
        }
        for condition in conditions
    }

    paired = {}
    condition_specs = result["answer_summary"]["conditions"]
    if result["answer_summary"]["source"] == "retrieval":
        for name, condition in condition_specs.items():
            if condition["chunker"] != "lumber":
                continue
            suffix = f"{condition['retriever']}__w{condition['evidence_words']}"
            for baseline in ("turn_packed", "word_packed"):
                baseline_name = f"{baseline}__{suffix}"
                if baseline_name not in conditions:
                    continue
                comparison = f"lumber_minus_{baseline}__{suffix}"
                paired[comparison] = {
                    "bertscore_f1": mean(
                        per_meeting[mid][name]["bertscore"]["f1"]
                        - per_meeting[mid][baseline_name]["bertscore"]["f1"]
                        for mid in meeting_ids
                    ),
                    "judge_mean": mean(
                        per_meeting[mid][name]["judge_mean"]
                        - per_meeting[mid][baseline_name]["judge_mean"]
                        for mid in meeting_ids
                    ),
                }

    return {
        "source": result["answer_summary"]["source"],
        "answer_model": result["answer_summary"]["answer_model"],
        "answer_count": len(records),
        "question_average": {
            condition: _metrics(
                [record for record in records if record["condition"] == condition]
            )
            for condition in conditions
        },
        "per_meeting": per_meeting,
        "meeting_average": meeting_average,
        "paired_meeting_average": paired,
    }


def evaluation_complete(
    saved: dict[str, object], expected: list[dict[str, object]]
) -> bool:
    records = saved.get("records")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        return False

    fields = ("meeting_id", "question_index", "condition")
    expected_keys = [
        tuple(record[field] for field in fields)
        for record in expected
    ]
    saved_keys = [
        tuple(record.get(field) for field in fields)
        for record in records
    ]
    return (
        saved_keys == expected_keys
        and len(saved_keys) == len(set(saved_keys))
        and all(
            set(record.get("bertscore", {})) == {"precision", "recall", "f1"}
            and record.get("judge", {}).get("score") in (1, 2, 3)
            for record in records
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    args = parser.parse_args()

    run = load_run_config(args.preset)
    spec = run.evaluation
    config = evaluation_config(run)
    pending = []
    completed = {}
    for answer_dir in (run.answers_dir / stage.name for stage in run.answers):
        answer_summary, records = load_stage(answer_dir, run.data_dir)
        inputs = {"answer_summary": answer_dir / "summary.json"}
        for meeting_id in answer_summary["meeting_ids"]:
            inputs[f"answers_{meeting_id}"] = answer_dir / f"{meeting_id}.json"
            inputs[f"meeting_{meeting_id}"] = run.data_dir / f"{meeting_id}.json"
        provenance = make_provenance(
            "evaluation", config, inputs, args.preset
        )
        output_path = run.evaluation_dir / f"{answer_dir.name}.json"
        if output_path.exists():
            try:
                saved = json.loads(output_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                saved = {}
            if (
                saved.get("provenance", {}).get("fingerprint")
                == provenance["fingerprint"]
                and evaluation_complete(saved, records)
            ):
                completed[answer_dir.name] = saved
                print(f"Evaluation {answer_dir.name}: existing", flush=True)
                continue
        pending.append(
            {
                "name": answer_dir.name,
                "output_path": output_path,
                "answer_summary": answer_summary,
                "records": records,
                "provenance": provenance,
            }
        )

    if pending:
        from bert_score import BERTScorer
        from huggingface_hub import snapshot_download
        import torch
        from meeting_qa_chunking.local_model import LocalChatModel

        model_path = snapshot_download(
            repo_id=spec.bertscore_model.name,
            revision=spec.bertscore_model.revision,
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
        )
        scorer = BERTScorer(
            model_type=model_path,
            num_layers=spec.bertscore_layers,
            device="cuda",
            rescale_with_baseline=False,
        )
        for stage in pending:
            add_bertscore(scorer, stage["records"], spec.bertscore_batch_size)
            print(f"BERTScore {stage['name']}: done", flush=True)
        del scorer
        gc.collect()
        torch.cuda.empty_cache()

        judge = LocalChatModel(
            model_name=spec.judge_model.name,
            revision=spec.judge_model.revision,
            max_new_tokens=spec.judge_max_new_tokens,
            seed=spec.judge_seed,
            temperature=spec.judge_temperature,
            cache_dir=Path(".cache/judgments"),
            prequantized=spec.judge_model.prequantized,
        )
        for stage in pending:
            calls_before = judge.model_calls
            hits_before = judge.cache_hits
            output_records = []
            for index, record in enumerate(stage["records"], start=1):
                response = judge(
                    build_judge_prompt(
                        record["question"],
                        record["reference_answer"],
                        record["gold_evidence"],
                        record["candidate_answer"],
                    )
                )
                score, reason = parse_judgment(response)
                output_records.append(
                    {
                        "meeting_id": record["meeting_id"],
                        "question_index": record["question_index"],
                        "condition": record["condition"],
                        "bertscore": record["bertscore"],
                        "judge": {
                            "score": score,
                            "reason": reason,
                            "raw_response": response,
                            "cache_hit": judge.last_cache_hit,
                        },
                    }
                )
                if index % 10 == 0 or index == len(stage["records"]):
                    print(
                        f"Judge {stage['name']}: {index}/{len(stage['records'])}",
                        flush=True,
                    )

            result = {
                "experiment_version": EXPERIMENT_VERSION,
                "provenance": stage["provenance"],
                "answer_summary": stage["answer_summary"],
                "judge_model_calls": judge.model_calls - calls_before,
                "judge_cache_hits": judge.cache_hits - hits_before,
                "records": output_records,
            }
            write_json(stage["output_path"], result)
            completed[stage["name"]] = result
            print(f"Evaluation {stage['name']}: saved", flush=True)

    summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "evaluation_config": config,
        "stages": {
            name: summarize_stage(result)
            for name, result in sorted(completed.items())
        },
    }
    write_json(run.evaluation_dir / "summary.json", summary)
    print(f"Evaluation summary: {run.evaluation_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
