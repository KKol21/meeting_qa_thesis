"""Run evidence-aware judging on the reproducible qualitative sample."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from tools.export_qualitative_workbook import (
    ANNOTATION_BUDGET,
    ANNOTATION_CHUNKERS,
    add_sampling_deltas,
    hydrate_evidence,
    index_records,
    load_cases,
    read_json,
    select_cases,
)
from tools.judge_grounded_answer import call_axis
from meeting_qa_chunking.artifacts import write_json
from meeting_qa_chunking.config import load_run_config
from meeting_qa_chunking.judging import (
    FAITHFULNESS_INSTRUCTION,
    SUFFICIENCY_INSTRUCTION,
    build_faithfulness_prompt,
    build_sufficiency_prompt,
    parse_axis_judgment,
)


def record_complete(record: dict) -> bool:
    try:
        aware = record["evidence_aware_judge"]
        parse_axis_judgment(
            aware["faithfulness"]["raw_response"], "unsupported_claims"
        )
        parse_axis_judgment(
            aware["evidence_sufficiency"]["raw_response"],
            "unavailable_reference_points",
        )
        return True
    except (KeyError, TypeError, ValueError):
        return False


def prepare(run, answer_stage: str, retriever: str, questions_per_type: int, seed: int):
    conditions = [
        f"{chunker}__{retriever}__w{ANNOTATION_BUDGET}"
        for chunker in ANNOTATION_CHUNKERS
    ]
    cases = load_cases(run, answer_stage, retriever)
    add_sampling_deltas(cases, retriever)
    selected = select_cases(cases, questions_per_type, seed)
    hydrate_evidence(run, selected, conditions)
    evaluation = index_records(
        read_json(run.evaluation_dir / f"{answer_stage}.json")["records"],
        "meeting_id",
        "question_index",
        "condition",
    )
    records = []
    for case in selected:
        for condition in conditions:
            result = case["results"][condition]
            records.append(
                {
                    "sample_id": case["sample_id"],
                    "meeting_id": case["meeting_id"],
                    "question_index": case["question_index"],
                    "query_type": case["query_type"],
                    "selection_reason": case["selection_reason"],
                    "condition": condition,
                    "question": case["question"],
                    "reference_answer": case["reference_answer"],
                    "gold_evidence": case["gold_evidence"],
                    "retrieved_evidence": result["retrieved_evidence"],
                    "retrieval_precision": result["precision"],
                    "retrieval_recall": result["recall"],
                    "candidate_answer": result["answer"],
                    "original_gold_only_judge": evaluation[
                        (case["meeting_id"], case["question_index"], condition)
                    ]["judge"],
                }
            )
    return conditions, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retriever", default="dense")
    parser.add_argument("--questions-per-type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answer-stage", default="retrieval-14b")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    run = load_run_config(args.preset)
    conditions, prepared = prepare(
        run, args.answer_stage, args.retriever, args.questions_per_type, args.seed
    )
    output = args.output or run.output_root / "grounded-judge" / "sample.json"
    saved = read_json(output) if output.exists() else {}
    existing = {
        (record["sample_id"], record["condition"]): record
        for record in saved.get("records", [])
        if record_complete(record)
    }
    records = [
        {
            **record,
            **existing.get((record["sample_id"], record["condition"]), {}),
        }
        for record in prepared
    ]

    spec = run.evaluation
    result = {
        "preset": str(args.preset),
        "sampling": {
            "questions_per_type": args.questions_per_type,
            "seed": args.seed,
            "retriever": args.retriever,
            "conditions": conditions,
        },
        "judge_config": {
            "model": asdict(spec.judge_model),
            "max_new_tokens": spec.judge_max_new_tokens,
            "temperature": spec.judge_temperature,
            "seed": spec.judge_seed,
            "faithfulness_prompt": FAITHFULNESS_INSTRUCTION,
            "sufficiency_prompt": SUFFICIENCY_INSTRUCTION,
        },
        "records": records,
    }
    write_json(output, result)
    if args.prepare_only:
        print(f"Prepared {len(records)} cases: {output}")
        return

    from meeting_qa_chunking.local_model import LocalChatModel

    judge = LocalChatModel(
        model_name=spec.judge_model.name,
        revision=spec.judge_model.revision,
        max_new_tokens=spec.judge_max_new_tokens,
        seed=spec.judge_seed,
        temperature=spec.judge_temperature,
        cache_dir=Path(".cache/grounded-judgments"),
        prequantized=spec.judge_model.prequantized,
    )
    for index, record in enumerate(records, 1):
        if record_complete(record):
            print(f"{index}/{len(records)} {record['sample_id']}: existing", flush=True)
            continue
        faithfulness = call_axis(
            judge,
            build_faithfulness_prompt(
                record["question"],
                record["retrieved_evidence"],
                record["candidate_answer"],
            ),
            "unsupported_claims",
        )
        sufficiency = call_axis(
            judge,
            build_sufficiency_prompt(
                record["question"],
                record["reference_answer"],
                record["retrieved_evidence"],
            ),
            "unavailable_reference_points",
        )
        record["evidence_aware_judge"] = {
            "faithfulness": faithfulness,
            "evidence_sufficiency": sufficiency,
        }
        write_json(output, result)
        print(f"{index}/{len(records)} {record['sample_id']}: saved", flush=True)
    print(f"Sample judgments: {output}")


if __name__ == "__main__":
    main()
