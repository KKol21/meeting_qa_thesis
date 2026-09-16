"""Compare final-chunk clipping policies using saved retrieval rankings."""

import argparse
from pathlib import Path
from statistics import mean

from meeting_qa_chunking.artifacts import read_json_object, write_json
from meeting_qa_chunking.evidence import Evidence, score_evidence, select_evidence
from meeting_qa_chunking.evidence_preparation import build_chunk_sets
from meeting_qa_chunking.config import load_run_config
from meeting_qa_chunking.qmsum import load_meeting


POLICIES = ("clip", "drop_partial", "expand_partial")
METRICS = ("precision", "recall", "f1", "zero_hit", "retrieved_words")


def complete_chunks(indices, chunks) -> Evidence:
    if not indices:
        return Evidence([])
    budget = sum(chunks[index].word_count for index in indices)
    return select_evidence([(index, 0.0) for index in indices], chunks, budget)


def variants(saved, chunks, budget) -> tuple[dict[str, Evidence], bool]:
    indices = saved["selected_chunk_indices"]
    clipped = select_evidence([(index, 0.0) for index in indices], chunks, budget)
    expanded = complete_chunks(indices, chunks)
    is_partial = expanded.word_count > clipped.word_count
    dropped = complete_chunks(indices[:-1], chunks) if is_partial else clipped
    return {
        "clip": clipped,
        "drop_partial": dropped,
        "expand_partial": expanded,
    }, is_partial


def evaluate(evidence, meeting, question, budget) -> dict[str, float]:
    score = score_evidence(evidence, meeting, question)
    precision, recall = score.precision, score.recall
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "zero_hit": float(recall == 0),
        "retrieved_words": score.retrieved_words,
        "budget_difference": score.retrieved_words - budget,
    }


def aggregate(rows, meeting_ids, conditions):
    output = {}
    for condition in conditions:
        output[condition] = {}
        for policy in POLICIES:
            per_meeting = {
                meeting_id: {
                    metric: mean(
                        row["policies"][policy][metric]
                        for row in rows
                        if row["meeting_id"] == meeting_id
                        and row["condition"] == condition
                    )
                    for metric in (*METRICS, "budget_difference")
                }
                for meeting_id in meeting_ids
            }
            output[condition][policy] = {
                metric: mean(item[metric] for item in per_meeting.values())
                for metric in (*METRICS, "budget_difference")
            }
        output[condition]["partial_rate"] = mean(
            mean(
                row["partial"]
                for row in rows
                if row["meeting_id"] == meeting_id
                and row["condition"] == condition
            )
            for meeting_id in meeting_ids
        )
    return output


def make_report(result) -> str:
    lines = [
        "# Evidence-clipping sensitivity analysis",
        "",
        f"**Scope:** {len(result['meeting_ids'])} meetings, "
        f"{result['question_count']} questions, {len(result['conditions'])} retrieval conditions.",
        "",
        "Rankings and chunk boundaries are held fixed. `clip` is the main exact-budget policy; "
        "`drop` stops before a partial final chunk; `expand` includes that chunk in full and may exceed the budget.",
        "",
        "Values are meeting-macro averages. Deltas are relative to `clip`.",
        "",
        "| Chunker | Retriever | Budget | Partial | Clip P/R/F1/Z | Drop dP/dR/dF1/dZ | Underfill | Expand dP/dR/dF1/dZ | Overflow |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, spec in result["conditions"].items():
        row = result["aggregate"][name]
        clip, drop, expand = (row[policy] for policy in POLICIES)
        lines.append(
            f"| {spec['chunker']} | {spec['retriever']} | {spec['evidence_words']} | "
            f"{row['partial_rate']:.1%} | "
            f"{clip['precision']:.3f}/{clip['recall']:.3f}/{clip['f1']:.3f}/{clip['zero_hit']:.1%} | "
            f"{drop['precision'] - clip['precision']:+.3f}/"
            f"{drop['recall'] - clip['recall']:+.3f}/"
            f"{drop['f1'] - clip['f1']:+.3f}/"
            f"{drop['zero_hit'] - clip['zero_hit']:+.1%} | "
            f"{-drop['budget_difference']:.0f} | "
            f"{expand['precision'] - clip['precision']:+.3f}/"
            f"{expand['recall'] - clip['recall']:+.3f}/"
            f"{expand['f1'] - clip['f1']:+.3f}/"
            f"{expand['zero_hit'] - clip['zero_hit']:+.1%} | "
            f"{max(0, expand['budget_difference']):.0f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- If the paired metric deltas are small, conclusions are insensitive to clipping.",
        "- `drop` preserves a hard budget but does not backfill from lower-ranked chunks; underfill reports its unused capacity.",
        "- `expand` preserves chunk integrity but its overflow makes evidence amounts unequal.",
        "- This analysis measures retrieval coverage only; it does not establish effects on generated answers.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = load_run_config(args.preset)
    meeting_ids = run.meeting_ids()
    rows = []
    conditions = None
    question_count = 0

    for meeting_id in meeting_ids:
        meeting = load_meeting(run.data_dir / f"{meeting_id}.json")
        question_count += len(meeting.questions)
        saved = read_json_object(run.retrieval_dir / f"{meeting_id}.json")
        conditions = conditions or saved["configurations"]
        if saved["configurations"] != conditions:
            raise ValueError("Retrieval configurations differ between meetings")
        chunk_sets = build_chunk_sets(
            meeting,
            run.lumber_dir / f"{meeting_id}.json",
            saved["chunking"]["turn_packed_max_words"],
            saved["chunking"]["word_packed_max_words"],
            {spec["chunker"] for spec in conditions.values()},
        )
        for question_index, question in enumerate(meeting.questions):
            saved_results = saved["questions"][question_index]["results"]
            for name, spec in conditions.items():
                policy_evidence, partial = variants(
                    saved_results[name],
                    chunk_sets[spec["chunker"]],
                    spec["evidence_words"],
                )
                policy_results = {
                    policy: evaluate(
                        evidence, meeting, question, spec["evidence_words"]
                    )
                    for policy, evidence in policy_evidence.items()
                }
                exact = policy_results["clip"]
                original = saved_results[name]
                if (
                    exact["retrieved_words"] != original["retrieved_words"]
                    or abs(exact["precision"] - original["precision"]) > 1e-12
                    or abs(exact["recall"] - original["recall"]) > 1e-12
                ):
                    raise ValueError(f"Could not reconstruct {meeting_id} {name}")
                rows.append(
                    {
                        "meeting_id": meeting_id,
                        "question_index": question_index,
                        "condition": name,
                        "partial": partial,
                        "policies": policy_results,
                    }
                )

    result = {
        "preset": str(args.preset),
        "meeting_ids": meeting_ids,
        "question_count": question_count,
        "conditions": conditions,
        "policies": list(POLICIES),
        "aggregate": aggregate(rows, meeting_ids, conditions),
        "questions": rows,
    }
    write_json(args.output, result)
    report = args.output.with_suffix(".md")
    report.write_text(make_report(result), encoding="utf-8")
    print(f"Sensitivity data: {args.output}")
    print(f"Sensitivity report: {report}")


if __name__ == "__main__":
    main()
