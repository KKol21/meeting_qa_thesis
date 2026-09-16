"""Small retrieval-only check of Lumber boundary-model sensitivity."""

import argparse
from pathlib import Path
from statistics import mean

from meeting_qa_chunking.artifacts import read_json_object, write_json
from meeting_qa_chunking.config import SEGMENTATION_MODELS, load_run_config
from meeting_qa_chunking.lumber import load_lumber_chunks
from meeting_qa_chunking.qmsum import load_meeting
from meeting_qa_chunking.uncertainty import Bootstrap
from tools.sweep_lumber_targets import METRICS, chunk_summary, evaluate


def boundaries(chunks_by_meeting):
    return {
        (meeting_id, chunk.end_turn + 1)
        for meeting_id, chunks in chunks_by_meeting.items()
        for chunk in chunks[:-1]
    }


def summarize(rows, meeting_ids, models, conditions, bootstrap, reference):
    per_meeting = {}
    aggregate = {}
    headline = {}
    for model in models:
        per_meeting[model] = {}
        for meeting_id in meeting_ids:
            selected = [row for row in rows if row["meeting_id"] == meeting_id]
            per_meeting[model][meeting_id] = {
                condition: {
                    metric: mean(
                        row["results"][model][condition][metric] for row in selected
                    )
                    for metric in METRICS
                }
                for condition in conditions
            }
        aggregate[model] = {
            condition: {
                metric: mean(
                    per_meeting[model][meeting_id][condition][metric]
                    for meeting_id in meeting_ids
                )
                for metric in METRICS
            }
            for condition in conditions
        }
        meeting_recall = [
            mean(
                per_meeting[model][meeting_id][condition]["recall"]
                for condition in conditions
            )
            for meeting_id in meeting_ids
        ]
        headline[model] = {
            "recall": bootstrap.mean(meeting_recall),
            "recall_minus_reference": None,
            "meeting_values": meeting_recall,
        }

    reference_values = headline[reference]["meeting_values"]
    for model in models:
        if model != reference:
            headline[model]["recall_minus_reference"] = bootstrap.paired(
                headline[model]["meeting_values"], reference_values
            )
        del headline[model]["meeting_values"]
    return per_meeting, aggregate, headline


def interval(value, signed=False):
    low, high = value["ci95"]
    pattern = "+.3f" if signed else ".3f"
    return (
        f"{format(value['mean'], pattern)} "
        f"[{format(low, pattern)}, {format(high, pattern)}]"
    )


def make_report(result):
    lines = [
        "# Lumber boundary-model check",
        "",
        f"**Scope:** {len(result['meeting_ids'])} QMSum validation meetings, "
        f"{result['question_count']} questions, target "
        f"{result['target_tokens']} pseudo-tokens.",
        "",
        "This retrieval-only diagnostic changes only the Lumber boundary model. Recall is first averaged across the nine retriever x evidence-budget environments within each meeting; meetings are then weighted equally and cluster-bootstrapped.",
        "",
        "> With only five meetings, use this to detect a large model effect or obvious failure—not to establish model equivalence.",
        "",
        "## Headline recall",
        "",
        f"Differences are relative to `{result['reference_model']}`.",
        "",
        "| Boundary model | Mean recall [95% CI] | Difference [95% CI] |",
        "|---|---:|---:|",
    ]
    for model, row in result["headline"].items():
        difference = row["recall_minus_reference"]
        lines.append(
            f"| {model} | {interval(row['recall'])} | "
            f"{interval(difference, True) if difference else 'reference'} |"
        )

    lines += [
        "",
        "## Chunk geometry",
        "",
        "| Boundary model | Chunks | Mean words | Median words | P90 words | Mean turns |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, row in result["chunking"].items():
        lines.append(
            f"| {model} | {row['chunk_count']} | "
            f"{row['words_per_chunk']['mean']:.1f} | "
            f"{row['words_per_chunk']['median']:.1f} | "
            f"{row['words_per_chunk']['p90']:.0f} | "
            f"{row['turns_per_chunk']['mean']:.1f} |"
        )

    lines += [
        "",
        "## Boundary agreement",
        "",
        "| Models | Shared boundaries | Jaccard |",
        "|---|---:|---:|",
    ]
    for row in result["boundary_overlap"]:
        lines.append(
            f"| {row['left']} / {row['right']} | {row['shared']} | "
            f"{row['jaccard']:.1%} |"
        )

    lines += [
        "",
        "## Retrieval cells",
        "",
        "| Model | Retriever | Budget | Precision | Recall | F1 | Zero-hit | MRR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, conditions in result["aggregate"].items():
        for condition, row in conditions.items():
            retriever, budget = condition.split("__w")
            lines.append(
                f"| {model} | {retriever} | {budget} | "
                f"{row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['f1']:.3f} | {row['zero_hit']:.1%} | {row['mrr']:.3f} |"
            )
    lines += [
        "",
        "A model is promising only if its paired recall difference is meaningfully positive without obtaining that gain merely through substantially larger chunks. Confirm any promising result on the complete validation set before changing the main experiment.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--segmentation-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=SEGMENTATION_MODELS, required=True)
    parser.add_argument("--reference-model", default="qwen2.5-14b")
    parser.add_argument("--meeting-count", type=int, default=5)
    parser.add_argument("--target-tokens", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = list(dict.fromkeys(args.models))
    if args.reference_model not in models:
        raise ValueError("reference-model must be included in models")
    if args.meeting_count <= 0 or args.target_tokens <= 0:
        raise ValueError("meeting-count and target-tokens must be positive")

    from meeting_qa_chunking.retrieval import (
        load_model,
        rank_chunks,
        rank_chunks_bm25,
        reciprocal_rank_fusion,
    )

    run = load_run_config(args.preset)
    meeting_ids = run.meeting_ids()[: args.meeting_count]
    if len(meeting_ids) != args.meeting_count:
        raise ValueError("meeting-count exceeds the preset meeting list")
    meetings = [load_meeting(run.data_dir / f"{name}.json") for name in meeting_ids]
    chunks = {model: {} for model in models}
    for model in models:
        for meeting in meetings:
            path = args.segmentation_root / model / f"{meeting.id}.json"
            saved = read_json_object(path)
            config = saved.get("provenance", {}).get("config", {})
            if config.get("model", {}).get("tag") != model:
                raise ValueError(f"{path} belongs to another model")
            if config.get("target_tokens") != args.target_tokens:
                raise ValueError(f"{path} records another target")
            chunks[model][meeting.id] = load_lumber_chunks(path, meeting)

    spec = run.retrieval
    retriever_model = load_model(spec.dense_model.name, spec.dense_model.revision)
    conditions = [
        f"{retriever}__w{budget}"
        for retriever in spec.retrievers
        for budget in spec.evidence_budgets
    ]
    rows = []
    for meeting in meetings:
        for question_index, question in enumerate(meeting.questions):
            results = {}
            for model in models:
                model_chunks = chunks[model][meeting.id]
                dense, _ = rank_chunks(
                    question.text,
                    model_chunks,
                    retriever_model,
                    spec.dense_model.name,
                    spec.dense_model.revision,
                )
                bm25 = rank_chunks_bm25(
                    question.text, model_chunks, spec.bm25_k1, spec.bm25_b
                )
                rankings = {
                    "dense": dense,
                    "bm25": bm25,
                    "hybrid": reciprocal_rank_fusion([dense, bm25], spec.rrf_k),
                }
                results[model] = {
                    f"{retriever}__w{budget}": evaluate(
                        question, meeting, model_chunks, rankings[retriever], budget
                    )
                    for retriever in spec.retrievers
                    for budget in spec.evidence_budgets
                }
            rows.append(
                {
                    "meeting_id": meeting.id,
                    "question_index": question_index,
                    "results": results,
                }
            )
        print(f"Retrieval {meeting.id}: done", flush=True)

    bootstrap = Bootstrap(len(meeting_ids), args.samples, args.seed)
    per_meeting, aggregate, headline = summarize(
        rows, meeting_ids, models, conditions, bootstrap, args.reference_model
    )
    boundary_sets = {model: boundaries(chunks[model]) for model in models}
    overlap = []
    for index, left in enumerate(models):
        for right in models[index + 1 :]:
            union = boundary_sets[left] | boundary_sets[right]
            overlap.append(
                {
                    "left": left,
                    "right": right,
                    "shared": len(boundary_sets[left] & boundary_sets[right]),
                    "jaccard": len(boundary_sets[left] & boundary_sets[right])
                    / len(union)
                    if union
                    else 1.0,
                }
            )

    result = {
        "preset": str(args.preset),
        "meeting_ids": meeting_ids,
        "question_count": len(rows),
        "models": models,
        "reference_model": args.reference_model,
        "target_tokens": args.target_tokens,
        "bootstrap_samples": args.samples,
        "chunking": {
            model: chunk_summary(
                [chunks[model][meeting_id] for meeting_id in meeting_ids],
                spec.evidence_budgets,
            )
            for model in models
        },
        "boundary_overlap": overlap,
        "headline": headline,
        "aggregate": aggregate,
        "per_meeting": per_meeting,
        "questions": rows,
    }
    write_json(args.output, result)
    report = args.output.with_suffix(".md")
    report.write_text(make_report(result), encoding="utf-8")
    print(f"Model check data: {args.output}")
    print(f"Model check report: {report}")


if __name__ == "__main__":
    main()
