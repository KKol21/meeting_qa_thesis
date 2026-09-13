"""Compare two Lumber window targets on the same QMSum meetings."""

import argparse
import json
import random
from pathlib import Path
from statistics import mean, stdev

from meeting_qa_chunking.artifacts import write_json
from meeting_qa_chunking.chunking import chunk_turn_packed, chunk_word_packed
from meeting_qa_chunking.config import load_run_config
from meeting_qa_chunking.evidence import (
    first_relevant_chunk_rank,
    score_evidence,
    select_evidence,
)
from meeting_qa_chunking.lumber import load_lumber_chunks
from meeting_qa_chunking.qmsum import load_meeting


TARGETS = (550, 1000)
RETRIEVERS = ("dense", "bm25", "hybrid")
METRICS = ("precision", "recall", "f1", "mrr")


def describe(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "mean": mean(ordered),
        "sd": stdev(ordered) if len(ordered) > 1 else 0.0,
        "median": percentile(0.5),
        "p90": percentile(0.9),
        "max": ordered[-1],
    }


def paired_statistics(
    differences: list[float], seed: int = 42, samples: int = 20_000
) -> dict[str, object]:
    """Meeting-paired bootstrap CI and two-sided sign-flip test."""

    observed = mean(differences)
    bootstrap_rng = random.Random(seed)
    bootstrap = sorted(
        mean(bootstrap_rng.choices(differences, k=len(differences)))
        for _ in range(samples)
    )
    sign_rng = random.Random(seed + 1)
    extreme = sum(
        abs(mean(value if sign_rng.random() < 0.5 else -value for value in differences))
        >= abs(observed)
        for _ in range(samples)
    )
    return {
        "mean_difference_1000_minus_550": observed,
        "bootstrap_95_ci": [
            bootstrap[int(0.025 * samples)],
            bootstrap[int(0.975 * samples)],
        ],
        "sign_flip_p": (extreme + 1) / (samples + 1),
    }


def evaluate_question(question, meeting, chunks, rankings, budget):
    result = {}
    for retriever, ranking in rankings.items():
        evidence = select_evidence(ranking, chunks, budget)
        score = score_evidence(evidence, meeting, question)
        first_rank = first_relevant_chunk_rank(ranking, chunks, question)
        precision, recall = score.precision, score.recall
        result[f"{retriever}__w{budget}"] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            "mrr": 1 / first_rank if first_rank else 0.0,
            "retrieved_words": score.retrieved_words,
            "gold_words": score.gold_words,
            "selected_chunk_indices": evidence.chunk_indices,
        }
    return result


def aggregate(questions, meeting_ids, conditions):
    aggregates = {}
    for condition in conditions:
        by_target = {}
        per_meeting = {}
        for target in TARGETS:
            per_meeting[target] = {
                meeting_id: {
                    metric: mean(
                        question["results"][str(target)][condition][metric]
                        for question in questions
                        if question["meeting_id"] == meeting_id
                    )
                    for metric in METRICS
                }
                for meeting_id in meeting_ids
            }
            by_target[str(target)] = {
                "question_average": {
                    metric: mean(
                        question["results"][str(target)][condition][metric]
                        for question in questions
                    )
                    for metric in METRICS
                },
                "meeting_average": {
                    metric: mean(
                        per_meeting[target][meeting_id][metric]
                        for meeting_id in meeting_ids
                    )
                    for metric in METRICS
                },
            }

        paired = {
            metric: paired_statistics(
                [
                    per_meeting[1000][meeting_id][metric]
                    - per_meeting[550][meeting_id][metric]
                    for meeting_id in meeting_ids
                ]
            )
            for metric in METRICS
        }
        recall_differences = [
            per_meeting[1000][meeting_id]["recall"]
            - per_meeting[550][meeting_id]["recall"]
            for meeting_id in meeting_ids
        ]
        paired["recall"]["meeting_wins_ties_losses"] = [
            sum(value > 1e-12 for value in recall_differences),
            sum(abs(value) <= 1e-12 for value in recall_differences),
            sum(value < -1e-12 for value in recall_differences),
        ]
        aggregates[condition] = {"targets": by_target, "paired": paired}
    return aggregates


def summarize_chunks(meeting_chunks):
    chunks = [chunk for group in meeting_chunks for chunk in group]
    return {
        "chunk_count": len(chunks),
        "chunks_per_meeting": describe([len(group) for group in meeting_chunks]),
        "words_per_chunk": describe([chunk.word_count for chunk in chunks]),
        "parts_per_chunk": describe([len(chunk.parts) for chunk in chunks]),
    }


def make_report(result: dict[str, object]) -> str:
    lines = [
        "# Lumber target-window comparison",
        "",
        "This is a paired sensitivity analysis on the 20 QMSum validation meetings used by the experiment. Both conditions use Qwen2.5-14B, greedy decoding, identical prompts, and identical retrieval settings; only `target_tokens` differs.",
        "",
        "## Chunk geometry",
        "",
        "| Method | Setting | Chunks | Mean chunks/meeting | Mean words/chunk | Median | P90 | Mean parts/chunk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (
        ("turn_packed_256", "turn-packed"),
        ("word_packed_256", "word-packed"),
    ):
        chunks = result["baselines"][name]
        lines.append(
            f"| {label} | 256 words | {chunks['chunk_count']} | "
            f"{chunks['chunks_per_meeting']['mean']:.1f} | "
            f"{chunks['words_per_chunk']['mean']:.1f} | "
            f"{chunks['words_per_chunk']['median']:.0f} | "
            f"{chunks['words_per_chunk']['p90']:.0f} | "
            f"{chunks['parts_per_chunk']['mean']:.1f} |"
        )
    for target in TARGETS:
        chunks = result["chunking"][str(target)]
        lines.append(
            f"| Lumber | {target} pseudo-tokens | {chunks['chunk_count']} | "
            f"{chunks['chunks_per_meeting']['mean']:.1f} | "
            f"{chunks['words_per_chunk']['mean']:.1f} | "
            f"{chunks['words_per_chunk']['median']:.0f} | "
            f"{chunks['words_per_chunk']['p90']:.0f} | "
            f"{chunks['parts_per_chunk']['mean']:.1f} |"
        )

    overlap = result["boundary_overlap"]
    lines += [
        "",
        "## Boundary stability",
        "",
        f"Across meetings, {overlap['shared']} exact boundaries are shared. This is {overlap['retained_from_550']:.1%} of 550-target boundaries and {overlap['retained_from_1000']:.1%} of 1,000-target boundaries; pooled Jaccard overlap is {overlap['jaccard']:.1%}. Mean meeting-level Jaccard overlap is {overlap['meeting_jaccard']['mean']:.1%}.",
        "",
        "Exact overlap is descriptive: sequential boundary decisions within a meeting are not independent.",
        "",
        "## Retrieval results",
        "",
        "Values are macro-averages over meetings.",
        "",
        "| Retriever | Evidence words | Target | Precision | Recall | F1 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, values in result["retrieval"].items():
        retriever, budget = condition.split("__w")
        for target in TARGETS:
            metrics = values["targets"][str(target)]["meeting_average"]
            lines.append(
                f"| {retriever} | {budget} | {target} | "
                + " | ".join(f"{metrics[name]:.3f}" for name in METRICS)
                + " |"
            )

    lines += [
        "",
        "## Paired difference in recall",
        "",
        "Differences are 1,000 minus 550. Confidence intervals bootstrap the 20 meetings; p-values use a two-sided meeting-level sign-flip test with 20,000 samples. These are exploratory and are not corrected for multiple comparisons.",
        "",
        "| Retriever | Evidence words | Mean difference | 95% CI | p | Meetings W/T/L |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, values in result["retrieval"].items():
        retriever, budget = condition.split("__w")
        paired = values["paired"]["recall"]
        low, high = paired["bootstrap_95_ci"]
        wins = "/".join(map(str, paired["meeting_wins_ties_losses"]))
        lines.append(
            f"| {retriever} | {budget} | {paired['mean_difference_1000_minus_550']:+.3f} | "
            f"[{low:+.3f}, {high:+.3f}] | {paired['sign_flip_p']:.3f} | {wins} |"
        )

    lines += [
        "",
        "## Recommendation for the final test",
        "",
        "Use **1,000 pseudo-tokens** as the final Lumber target and retain 550 as the method-faithfulness sensitivity condition.",
        "",
        "The main reason is comparability. The 1,000-target median is 243 words, close to the turn-packed and word-packed medians of 240 and 256; the 550-target median is only 170. Using 550 would therefore confound semantic segmentation with substantially finer chunk granularity. Compare the retrieval estimates across every configured retriever/budget combination before freezing the target.",
        "",
        "The statistical evidence is exploratory. Interpret individual intervals and uncorrected p-values cautiously because several correlated retrieval comparisons are reported. The recommendation should rest on matched granularity and consistent behavior, not one isolated significance result.",
        "",
        "Keep the configured 512-, 1,024-, and 2,048-word retrieved-evidence budgets in the final test. They expose the precision/recall tradeoff from median-scale through near-P90 gold evidence size.",
        "",
        "This choice is made on validation data and should now be frozen before any held-out test run.",
        "",
        "## Methodological limitations",
        "",
        "- The comparison uses validation data and is intended for parameter selection, not final performance reporting.",
        "- `target_tokens` uses Lumber's approximate rendered-word accounting, not the Qwen tokenizer.",
        "- Gold relevance is turn-level, so every word in a gold turn counts as relevant.",
        "- MRR rewards the first chunk overlapping any gold turn and can favor larger chunks.",
        "- Multiple correlated retrieval comparisons are reported; inferential statistics are exploratory.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--segmentation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
    chunk_values = {target: [] for target in TARGETS}
    baseline_values = {"turn_packed_256": [], "word_packed_256": []}
    boundary_rows = []
    questions = []

    for meeting_id in meeting_ids:
        meeting = load_meeting(run.data_dir / f"{meeting_id}.json")
        chunk_sets = {
            target: load_lumber_chunks(
                args.segmentation_root / str(target) / f"{meeting_id}.json",
                meeting,
            )
            for target in TARGETS
        }
        baseline_values["turn_packed_256"].append(
            chunk_turn_packed(meeting.turns, spec.turn_packed_max_words)
        )
        baseline_values["word_packed_256"].append(
            chunk_word_packed(meeting.turns, spec.word_packed_max_words)
        )
        boundaries = {
            target: {chunk.end_turn + 1 for chunk in chunks[:-1]}
            for target, chunks in chunk_sets.items()
        }
        shared = boundaries[550] & boundaries[1000]
        boundary_rows.append(
            {
                "meeting_id": meeting_id,
                "boundaries_550": len(boundaries[550]),
                "boundaries_1000": len(boundaries[1000]),
                "shared": len(shared),
                "jaccard": len(shared) / len(boundaries[550] | boundaries[1000]),
            }
        )
        for target, chunks in chunk_sets.items():
            chunk_values[target].append(chunks)

        for question_index, question in enumerate(meeting.questions):
            row = {
                "meeting_id": meeting_id,
                "question_index": question_index,
                "question": question.text,
                "results": {},
            }
            for target, chunks in chunk_sets.items():
                dense, _cache_hit = rank_chunks(
                    question.text,
                    chunks,
                    model,
                    spec.dense_model.name,
                    spec.dense_model.revision,
                )
                bm25 = rank_chunks_bm25(
                    question.text, chunks, spec.bm25_k1, spec.bm25_b
                )
                rankings = {
                    "dense": dense,
                    "bm25": bm25,
                    "hybrid": reciprocal_rank_fusion([dense, bm25], spec.rrf_k),
                }
                row["results"][str(target)] = {}
                for budget in spec.evidence_budgets:
                    row["results"][str(target)].update(
                        evaluate_question(question, meeting, chunks, rankings, budget)
                    )
            questions.append(row)
        print(f"Compared {meeting_id}", flush=True)

    chunking = {
        str(target): summarize_chunks(meeting_chunks)
        for target, meeting_chunks in chunk_values.items()
    }

    shared = sum(row["shared"] for row in boundary_rows)
    count_550 = sum(row["boundaries_550"] for row in boundary_rows)
    count_1000 = sum(row["boundaries_1000"] for row in boundary_rows)
    conditions = [
        f"{retriever}__w{budget}"
        for retriever in RETRIEVERS
        for budget in spec.evidence_budgets
    ]
    result = {
        "preset": str(args.preset),
        "meeting_ids": meeting_ids,
        "question_count": len(questions),
        "targets": list(TARGETS),
        "dense_model": {
            "name": spec.dense_model.name,
            "revision": spec.dense_model.revision,
        },
        "evidence_budgets": spec.evidence_budgets,
        "baselines": {
            name: summarize_chunks(meeting_chunks)
            for name, meeting_chunks in baseline_values.items()
        },
        "chunking": chunking,
        "boundary_overlap": {
            "shared": shared,
            "retained_from_550": shared / count_550,
            "retained_from_1000": shared / count_1000,
            "jaccard": shared / (count_550 + count_1000 - shared),
            "meeting_jaccard": describe([row["jaccard"] for row in boundary_rows]),
            "meetings": boundary_rows,
        },
        "retrieval": aggregate(questions, meeting_ids, conditions),
        "questions": questions,
    }
    write_json(args.output, result)
    args.output.with_suffix(".md").write_text(make_report(result), encoding="utf-8")
    print(f"Comparison: {args.output}")
    print(f"Report: {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
