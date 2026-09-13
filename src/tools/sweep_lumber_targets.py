"""Evaluate a small Lumber target-window sweep on QMSum."""

import argparse
from pathlib import Path
from statistics import mean, median, stdev

from meeting_qa_chunking.artifacts import read_json_object, write_json
from meeting_qa_chunking.chunking import chunk_turn_packed, chunk_word_packed
from meeting_qa_chunking.config import load_run_config
from meeting_qa_chunking.evidence import (
    first_relevant_chunk_rank,
    score_evidence,
    select_evidence,
)
from meeting_qa_chunking.lumber import load_lumber_chunks
from meeting_qa_chunking.qmsum import load_meeting


METRICS = (
    "precision",
    "recall",
    "f1",
    "mrr",
    "zero_hit",
    "clipped",
    "selected_chunks",
    "final_chunk_fraction",
)


def describe(values) -> dict[str, float]:
    values = list(values)
    ordered = sorted(values)
    return {
        "mean": mean(values),
        "sd": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "p90": ordered[round(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def chunk_summary(meeting_chunks, budgets) -> dict[str, object]:
    chunks = [chunk for group in meeting_chunks for chunk in group]
    return {
        "chunk_count": len(chunks),
        "chunks_per_meeting": describe(len(group) for group in meeting_chunks),
        "words_per_chunk": describe(chunk.word_count for chunk in chunks),
        "turns_per_chunk": describe(len(chunk.parts) for chunk in chunks),
        "chunks_over_budget": {
            str(budget): sum(chunk.word_count > budget for chunk in chunks)
            / len(chunks)
            for budget in budgets
        },
    }


def evaluate(question, meeting, chunks, ranking, budget) -> dict[str, object]:
    evidence = select_evidence(ranking, chunks, budget)
    score = score_evidence(evidence, meeting, question)
    first_rank = first_relevant_chunk_rank(ranking, chunks, question)
    selected = evidence.chunk_indices
    selected_words = {
        index: sum(
            part.word_count for part in evidence.parts if part.chunk_index == index
        )
        for index in selected
    }
    final_fraction = (
        selected_words[selected[-1]] / chunks[selected[-1]].word_count
        if selected
        else 0.0
    )
    precision, recall = score.precision, score.recall
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "mrr": 1 / first_rank if first_rank else 0.0,
        "zero_hit": float(recall == 0),
        "clipped": float(final_fraction < 1),
        "selected_chunks": len(selected),
        "final_chunk_fraction": final_fraction,
        "retrieved_words": score.retrieved_words,
        "gold_words": score.gold_words,
    }


def aggregate(questions, meeting_ids, targets, conditions):
    result = {}
    for condition in conditions:
        result[condition] = {}
        for target in targets:
            per_meeting = {
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
            result[condition][str(target)] = {
                "question_average": {
                    metric: mean(
                        question["results"][str(target)][condition][metric]
                        for question in questions
                    )
                    for metric in METRICS
                },
                "meeting_average": {
                    metric: mean(values[metric] for values in per_meeting.values())
                    for metric in METRICS
                },
                "meeting_sd": {
                    metric: (
                        stdev(values[metric] for values in per_meeting.values())
                        if len(per_meeting) > 1
                        else 0.0
                    )
                    for metric in METRICS
                },
            }
    return result


def select_target(targets, chunking, baselines, retrieval):
    baseline_median = mean(
        baseline["words_per_chunk"]["median"] for baseline in baselines.values()
    )
    diagnostics = {}
    for target in targets:
        target_results = [condition[str(target)] for condition in retrieval.values()]
        chunk_median = chunking[str(target)]["words_per_chunk"]["median"]
        diagnostics[str(target)] = {
            "median_chunk_words": chunk_median,
            "distance_from_baseline_median": abs(chunk_median - baseline_median),
            "mean_recall_across_conditions": mean(
                result["meeting_average"]["recall"] for result in target_results
            ),
            "mean_f1_across_conditions": mean(
                result["meeting_average"]["f1"] for result in target_results
            ),
            "mean_zero_hit_across_conditions": mean(
                result["meeting_average"]["zero_hit"] for result in target_results
            ),
        }
    best_recall = max(
        item["mean_recall_across_conditions"] for item in diagnostics.values()
    )
    eligible = [
        target
        for target in targets
        if diagnostics[str(target)]["mean_recall_across_conditions"]
        >= best_recall - 0.01
    ]
    chosen = min(
        eligible,
        key=lambda target: (
            diagnostics[str(target)]["distance_from_baseline_median"],
            -diagnostics[str(target)]["mean_f1_across_conditions"],
        ),
    )
    return {
        "rule": (
            "Among targets within 0.01 absolute mean recall of the best target, "
            "choose the median chunk length closest to the two baseline medians."
        ),
        "baseline_median_midpoint": baseline_median,
        "recommended_target": chosen,
        "diagnostics": diagnostics,
    }


def make_report(result) -> str:
    targets = result["targets"]
    lines = [
        "# Lumber target-window sweep",
        "",
        f"**Scope:** {len(result['meeting_ids'])} QMSum validation meetings, "
        f"{result['question_count']} questions; targets "
        + ", ".join(map(str, targets))
        + " pseudo-tokens.",
        "",
        "Only the Lumber target changes. Boundary model, prompt, decoding, retrieval, and evidence budgets are fixed.",
        "",
        "## Chunk geometry",
        "",
        "| Target | Chunks | Mean/meeting | Mean words | Median words | P90 words | Mean turns |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in targets:
        row = result["chunking"][str(target)]
        lines.append(
            f"| {target} | {row['chunk_count']} | "
            f"{row['chunks_per_meeting']['mean']:.1f} | "
            f"{row['words_per_chunk']['mean']:.1f} | "
            f"{row['words_per_chunk']['median']:.1f} | "
            f"{row['words_per_chunk']['p90']:.0f} | "
            f"{row['turns_per_chunk']['mean']:.1f} |"
        )

    lines += [
        "",
        "## Adjacent-target boundary overlap",
        "",
        "| Targets | Shared | Retained from smaller | Retained from larger | Jaccard |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["boundary_overlap"]:
        lines.append(
            f"| {row['smaller']} / {row['larger']} | {row['shared']} | "
            f"{row['retained_from_smaller']:.1%} | "
            f"{row['retained_from_larger']:.1%} | {row['jaccard']:.1%} |"
        )

    lines += [
        "",
        "## Retrieval",
        "",
        "Values are macro-averaged over meetings. Zero-hit and clipping are question proportions. Clipping means the evidence budget cut the final selected chunk; final fraction is the retained share of that chunk.",
        "",
        "| Retriever | Budget | Target | Precision | Recall | F1 | Zero-hit | Clipping | Final fraction | Selected chunks | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, target_results in result["retrieval"].items():
        retriever, budget = condition.split("__w")
        for target in targets:
            row = target_results[str(target)]["meeting_average"]
            lines.append(
                f"| {retriever} | {budget} | {target} | "
                f"{row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['f1']:.3f} | {row['zero_hit']:.1%} | "
                f"{row['clipped']:.1%} | {row['final_chunk_fraction']:.2f} | "
                f"{row['selected_chunks']:.2f} | {row['mrr']:.3f} |"
            )

    selection = result["selection"]
    lines += [
        "",
        "## Target-selection diagnostics",
        "",
        f"Predeclared rule: {selection['rule']}",
        "",
        "| Target | Median words | Baseline distance | Mean recall | Mean F1 | Mean zero-hit |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for target in targets:
        row = selection["diagnostics"][str(target)]
        lines.append(
            f"| {target} | {row['median_chunk_words']:.1f} | "
            f"{row['distance_from_baseline_median']:.1f} | "
            f"{row['mean_recall_across_conditions']:.3f} | "
            f"{row['mean_f1_across_conditions']:.3f} | "
            f"{row['mean_zero_hit_across_conditions']:.1%} |"
        )
    lines += [
        "",
        f"**Rule-based recommendation:** `{selection['recommended_target']}` pseudo-tokens.",
        "",
        "Freeze the selected target before testing. Keep 512, 1,024, and 2,048 evidence words in the final ablation; they represent roughly 2, 4, and 8 strict chunks and expose the recall/context tradeoff.",
        "",
        "## Notes",
        "",
        "- `target_tokens` is Lumber's rendered-word approximation, not tokenizer tokens.",
        "- Relevance is turn-level, so all words in a gold turn count as relevant.",
        "- Larger chunks can inflate MRR by increasing overlap probability.",
        "- The selection score averages correlated retrieval conditions and is a transparent decision aid, not a significance test.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--segmentation-root", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from meeting_qa_chunking.retrieval import (
        load_model,
        rank_chunks,
        rank_chunks_bm25,
        reciprocal_rank_fusion,
    )

    targets = sorted(set(args.targets))
    if len(targets) < 2 or any(target <= 0 for target in targets):
        raise ValueError("Provide at least two distinct positive targets")

    run = load_run_config(args.preset)
    spec = run.retrieval
    meeting_ids = run.meeting_ids()
    meetings = [
        load_meeting(run.data_dir / f"{meeting_id}.json")
        for meeting_id in meeting_ids
    ]
    model = load_model(spec.dense_model.name, spec.dense_model.revision)
    print(f"Embedding device: {model.device}", flush=True)

    chunks_by_target = {target: {} for target in targets}
    boundary_sets = {target: set() for target in targets}
    for target in targets:
        for meeting in meetings:
            path = args.segmentation_root / str(target) / f"{meeting.id}.json"
            saved = read_json_object(path)
            configured_target = (
                saved.get("provenance", {}).get("config", {}).get("target_tokens")
            )
            if configured_target != target:
                raise ValueError(
                    f"{path} records target_tokens={configured_target}, expected {target}"
                )
            chunks = load_lumber_chunks(path, meeting)
            chunks_by_target[target][meeting.id] = chunks
            boundary_sets[target].update(
                (meeting.id, chunk.end_turn + 1) for chunk in chunks[:-1]
            )

    baselines = {
        "turn_packed": chunk_summary(
            [
                chunk_turn_packed(meeting.turns, spec.turn_packed_max_words)
                for meeting in meetings
            ],
            spec.evidence_budgets,
        ),
        "word_packed": chunk_summary(
            [
                chunk_word_packed(meeting.turns, spec.word_packed_max_words)
                for meeting in meetings
            ],
            spec.evidence_budgets,
        ),
    }
    chunking = {
        str(target): chunk_summary(
            [chunks_by_target[target][meeting.id] for meeting in meetings],
            spec.evidence_budgets,
        )
        for target in targets
    }

    questions = []
    for meeting in meetings:
        for question_index, question in enumerate(meeting.questions):
            target_results = {}
            for target in targets:
                chunks = chunks_by_target[target][meeting.id]
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
                target_results[str(target)] = {
                    f"{retriever}__w{budget}": evaluate(
                        question, meeting, chunks, rankings[retriever], budget
                    )
                    for retriever in spec.retrievers
                    for budget in spec.evidence_budgets
                }
            questions.append(
                {
                    "meeting_id": meeting.id,
                    "question_index": question_index,
                    "question": question.text,
                    "results": target_results,
                }
            )
            print(f"Retrieval {meeting.id} question {question_index}", flush=True)

    conditions = [
        f"{retriever}__w{budget}"
        for retriever in spec.retrievers
        for budget in spec.evidence_budgets
    ]
    retrieval = aggregate(questions, meeting_ids, targets, conditions)
    boundary_overlap = []
    for smaller, larger in zip(targets, targets[1:]):
        left, right = boundary_sets[smaller], boundary_sets[larger]
        shared = len(left & right)
        boundary_overlap.append(
            {
                "smaller": smaller,
                "larger": larger,
                "shared": shared,
                "retained_from_smaller": shared / len(left) if left else 0.0,
                "retained_from_larger": shared / len(right) if right else 0.0,
                "jaccard": shared / len(left | right) if left or right else 1.0,
            }
        )

    result = {
        "preset": str(args.preset),
        "meeting_ids": meeting_ids,
        "question_count": len(questions),
        "targets": targets,
        "evidence_budgets": list(spec.evidence_budgets),
        "retrievers": list(spec.retrievers),
        "dense_model": {
            "name": spec.dense_model.name,
            "revision": spec.dense_model.revision,
        },
        "baselines": baselines,
        "chunking": chunking,
        "boundary_overlap": boundary_overlap,
        "retrieval": retrieval,
        "selection": select_target(targets, chunking, baselines, retrieval),
        "questions": questions,
    }
    write_json(args.output, result)
    report_path = args.output.with_suffix(".md")
    report_path.write_text(make_report(result), encoding="utf-8")
    print(f"Sweep data: {args.output}")
    print(f"Sweep report: {report_path}")


if __name__ == "__main__":
    main()
