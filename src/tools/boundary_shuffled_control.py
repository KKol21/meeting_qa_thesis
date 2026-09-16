"""Compare Lumber retrieval with geometry-matched random boundaries."""

import argparse
from pathlib import Path
import random
from statistics import mean, median

from meeting_qa_chunking.artifacts import read_json_object, write_json
from meeting_qa_chunking.chunking import Chunk, turn_word_count
from meeting_qa_chunking.config import load_run_config
from meeting_qa_chunking.lumber import load_lumber_chunks
from meeting_qa_chunking.qmsum import load_meeting
from meeting_qa_chunking.uncertainty import bootstrap_mean
from tools.sweep_lumber_targets import evaluate


METRICS = ("precision", "recall", "f1", "zero_hit", "mrr")


def boundaries(chunks) -> set[int]:
    return {chunk.end_turn + 1 for chunk in chunks[:-1]}


def shuffled_chunks(meeting, lumber, seed, candidates=1000):
    """Choose a random legal partition closely matching Lumber's geometry."""

    chunk_count = len(lumber)
    turn_count = len(meeting.turns)
    if not 1 < chunk_count <= turn_count:
        raise ValueError("Boundary shuffling requires two or more legal chunks")

    prefix = [0]
    for turn in meeting.turns:
        prefix.append(prefix[-1] + turn_word_count(turn))
    target_words = sorted(chunk.word_count for chunk in lumber)
    target_turns = [len(chunk.parts) for chunk in lumber]
    actual = boundaries(lumber)
    rng = random.Random(f"{seed}:{meeting.id}")
    best = None

    for _ in range(candidates):
        shuffled_turns = target_turns.copy()
        rng.shuffle(shuffled_turns)
        points = [0]
        for length in shuffled_turns:
            points.append(points[-1] + length)
        if set(points[1:-1]) == actual:
            continue
        words = sorted(
            prefix[end] - prefix[start]
            for start, end in zip(points, points[1:])
        )
        distance = (
            sum(abs(left - right) for left, right in zip(words, target_words))
            / prefix[-1]
        )
        if best is None or distance < best[0]:
            best = distance, points

    if best is None:
        raise ValueError("Could not create a shuffled boundary control")
    distance, points = best
    return (
        [
            Chunk.from_turns(index, meeting.turns[start:end])
            for index, (start, end) in enumerate(zip(points, points[1:]))
        ],
        distance,
    )


def average(rows, metric):
    return mean(row[metric] for row in rows)


def summarize(rows, meeting_ids, conditions):
    output = {}
    for condition in conditions:
        differences = {metric: [] for metric in METRICS}
        actual_meetings = {metric: [] for metric in METRICS}
        control_meetings = {metric: [] for metric in METRICS}
        for meeting_id in meeting_ids:
            selected = [
                row for row in rows
                if row["meeting_id"] == meeting_id
                and row["condition"] == condition
            ]
            for metric in METRICS:
                actual = average((row["actual"] for row in selected), metric)
                control = mean(
                    result[metric]
                    for row in selected
                    for result in row["controls"]
                )
                actual_meetings[metric].append(actual)
                control_meetings[metric].append(control)
                differences[metric].append(actual - control)

        output[condition] = {
            "actual": {
                metric: mean(actual_meetings[metric]) for metric in METRICS
            },
            "control": {
                metric: mean(control_meetings[metric]) for metric in METRICS
            },
            "actual_minus_control": {},
        }
        for metric, values in differences.items():
            output[condition]["actual_minus_control"][metric] = bootstrap_mean(
                values
            )
    return output


def geometry_summary(actual_groups, controls, overlaps, distances):
    actual = [chunk for group in actual_groups for chunk in group]
    shuffled = [chunk for versions in controls for group in versions for chunk in group]
    return {
        "chunk_count_preserved": all(
            len(group) == len(actual_groups[index])
            for versions in controls
            for index, group in enumerate(versions)
        ),
        "actual_mean_words": mean(chunk.word_count for chunk in actual),
        "control_mean_words": mean(chunk.word_count for chunk in shuffled),
        "actual_median_words": median(chunk.word_count for chunk in actual),
        "control_median_words": median(chunk.word_count for chunk in shuffled),
        "actual_mean_turns": mean(len(chunk.parts) for chunk in actual),
        "control_mean_turns": mean(len(chunk.parts) for chunk in shuffled),
        "mean_boundary_retention": mean(overlaps),
        "mean_matching_distance": mean(distances),
    }


def interval(value) -> str:
    low, high = value["ci95"]
    return f"{value['mean']:+.3f} [{low:+.3f}, {high:+.3f}]"


def make_report(result) -> str:
    geometry = result["geometry"]
    lines = [
        "# Boundary-shuffled Lumber control",
        "",
        f"**Scope:** {len(result['meeting_ids'])} QMSum validation meetings, "
        f"{result['question_count']} questions, Lumber target {result['target']} pseudo-tokens, "
        f"{result['randomizations']} random partitions per meeting.",
        "",
        "Each control permutes Lumber's exact turns-per-chunk distribution across legal turn boundaries. From random permutations, the closest word-size distribution is retained; semantic boundary locations are not used.",
        "",
        "## Geometry check",
        "",
        "| Chunks preserved | Lumber mean/median words | Control mean/median words | Lumber/control mean turns | Lumber boundaries retained |",
        "|---:|---:|---:|---:|---:|",
        f"| {geometry['chunk_count_preserved']} | "
        f"{geometry['actual_mean_words']:.1f}/{geometry['actual_median_words']:.1f} | "
        f"{geometry['control_mean_words']:.1f}/{geometry['control_median_words']:.1f} | "
        f"{geometry['actual_mean_turns']:.1f}/{geometry['control_mean_turns']:.1f} | "
        f"{geometry['mean_boundary_retention']:.1%} |",
        "",
        "## Retrieval comparison",
        "",
        "Values are meeting-macro averages. Differences are actual Lumber minus the mean shuffled control; intervals are paired 95% bootstrap intervals across meetings.",
        "",
        "| Retriever | Budget | Lumber P/R/F1/Z | Control P/R/F1/Z | dRecall [95% CI] | dF1 [95% CI] | dMRR [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, spec in result["conditions"].items():
        row = result["aggregate"][name]
        actual, control = row["actual"], row["control"]
        delta = row["actual_minus_control"]
        lines.append(
            f"| {spec['retriever']} | {spec['evidence_words']} | "
            f"{actual['precision']:.3f}/{actual['recall']:.3f}/{actual['f1']:.3f}/{actual['zero_hit']:.1%} | "
            f"{control['precision']:.3f}/{control['recall']:.3f}/{control['f1']:.3f}/{control['zero_hit']:.1%} | "
            f"{interval(delta['recall'])} | {interval(delta['f1'])} | "
            f"{interval(delta['mrr'])} |"
        )
    lines += [
        "",
        "## Interpretation rule",
        "",
        "Actual Lumber provides direct evidence for semantic boundary placement when its paired recall or F1 interval lies above zero. MRR remains secondary because overlap probability is size-sensitive, even though geometry is matched here.",
        "",
        "This is a retrieval-only secondary ablation; no answers are generated.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--segmentation-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--randomizations", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.randomizations < 2 or args.candidates <= 0:
        raise ValueError("Use at least two randomizations and one candidate")

    from meeting_qa_chunking.retrieval import (
        load_model,
        rank_chunks,
        rank_chunks_bm25,
        reciprocal_rank_fusion,
    )

    run = load_run_config(args.preset)
    spec = run.retrieval
    meeting_ids = run.meeting_ids()
    model = load_model(spec.dense_model.name, spec.dense_model.revision)
    rows = []
    actual_groups = []
    controls_by_version = [[] for _ in range(args.randomizations)]
    overlaps, distances = [], []
    question_count = 0
    conditions = {
        f"{retriever}__w{budget}": {
            "retriever": retriever,
            "evidence_words": budget,
        }
        for retriever in spec.retrievers
        for budget in spec.evidence_budgets
    }

    for meeting_id in meeting_ids:
        meeting = load_meeting(run.data_dir / f"{meeting_id}.json")
        segmentation_path = args.segmentation_dir / f"{meeting_id}.json"
        segmentation = read_json_object(segmentation_path)
        configured_target = (
            segmentation.get("provenance", {}).get("config", {}).get("target_tokens")
        )
        if configured_target != args.target:
            raise ValueError(
                f"{segmentation_path} records target_tokens={configured_target}, "
                f"expected {args.target}"
            )
        lumber = load_lumber_chunks(segmentation_path, meeting)
        actual_groups.append(lumber)
        question_count += len(meeting.questions)
        versions = []
        actual_boundaries = boundaries(lumber)
        for version in range(args.randomizations):
            control, distance = shuffled_chunks(
                meeting, lumber, args.seed + version, args.candidates
            )
            versions.append(control)
            controls_by_version[version].append(control)
            control_boundaries = boundaries(control)
            overlaps.append(
                len(actual_boundaries & control_boundaries)
                / len(actual_boundaries)
            )
            distances.append(distance)

        for question_index, question in enumerate(meeting.questions):
            dense, _cache_hit = rank_chunks(
                question.text,
                lumber,
                model,
                spec.dense_model.name,
                spec.dense_model.revision,
            )
            bm25 = rank_chunks_bm25(
                question.text, lumber, spec.bm25_k1, spec.bm25_b
            )
            actual_rankings = {
                "dense": dense,
                "bm25": bm25,
                "hybrid": reciprocal_rank_fusion([dense, bm25], spec.rrf_k),
            }
            for name, condition in conditions.items():
                rows.append(
                    {
                        "meeting_id": meeting_id,
                        "question_index": question_index,
                        "condition": name,
                        "actual": evaluate(
                            question,
                            meeting,
                            lumber,
                            actual_rankings[condition["retriever"]],
                            condition["evidence_words"],
                        ),
                        "controls": [],
                    }
                )
            question_rows = rows[-len(conditions):]
            for control in versions:
                dense, _cache_hit = rank_chunks(
                    question.text,
                    control,
                    model,
                    spec.dense_model.name,
                    spec.dense_model.revision,
                )
                bm25 = rank_chunks_bm25(
                    question.text, control, spec.bm25_k1, spec.bm25_b
                )
                rankings = {
                    "dense": dense,
                    "bm25": bm25,
                    "hybrid": reciprocal_rank_fusion([dense, bm25], spec.rrf_k),
                }
                for row in question_rows:
                    condition = conditions[row["condition"]]
                    row["controls"].append(
                        evaluate(
                            question,
                            meeting,
                            control,
                            rankings[condition["retriever"]],
                            condition["evidence_words"],
                        )
                    )
            print(f"Control retrieval {meeting_id} question {question_index}", flush=True)

    result = {
        "preset": str(args.preset),
        "segmentation_dir": str(args.segmentation_dir),
        "target": args.target,
        "meeting_ids": meeting_ids,
        "question_count": question_count,
        "randomizations": args.randomizations,
        "candidates_per_randomization": args.candidates,
        "seed": args.seed,
        "conditions": conditions,
        "geometry": geometry_summary(
            actual_groups, controls_by_version, overlaps, distances
        ),
        "aggregate": summarize(rows, meeting_ids, conditions),
        "questions": rows,
    }
    write_json(args.output, result)
    report = args.output.with_suffix(".md")
    report.write_text(make_report(result), encoding="utf-8")
    print(f"Control data: {args.output}")
    print(f"Control report: {report}")


if __name__ == "__main__":
    main()
