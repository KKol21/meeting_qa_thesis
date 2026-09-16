"""Retrieval-only chunk-size sweep for deterministic packed baselines."""

import argparse
from pathlib import Path
from statistics import mean, median

from meeting_qa_chunking.artifacts import write_json
from meeting_qa_chunking.chunking import chunk_turn_packed, chunk_word_packed
from meeting_qa_chunking.config import load_run_config
from meeting_qa_chunking.qmsum import load_meeting
from tools.sweep_lumber_targets import METRICS, evaluate


CHUNKERS = {
    "turn_packed": chunk_turn_packed,
    "word_packed": chunk_word_packed,
}


def chunk_summary(groups) -> dict[str, float]:
    chunks = [chunk for group in groups for chunk in group]
    words = [chunk.word_count for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "chunks_per_meeting": mean(len(group) for group in groups),
        "mean_words": mean(words),
        "median_words": median(words),
        "mean_turns": mean(
            len({part.turn_id for part in chunk.parts}) for chunk in chunks
        ),
    }


def aggregate(questions, meeting_ids, conditions):
    output = {}
    for name in conditions:
        per_meeting = {
            meeting_id: {
                metric: mean(
                    question["results"][name][metric]
                    for question in questions
                    if question["meeting_id"] == meeting_id
                )
                for metric in METRICS
            }
            for meeting_id in meeting_ids
        }
        output[name] = {
            metric: mean(row[metric] for row in per_meeting.values())
            for metric in METRICS
        }
    return output


def make_report(result) -> str:
    lines = [
        "# Deterministic chunk-size sensitivity",
        "",
        f"**Scope:** {len(result['meeting_ids'])} QMSum validation meetings, "
        f"{result['question_count']} questions; chunk sizes "
        + ", ".join(map(str, result["sizes"]))
        + " words.",
        "",
        "Only deterministic packed-chunk size changes. Retrieval methods and evidence budgets match the main experiment.",
        "",
        "## Chunk geometry",
        "",
        "| Chunker | Size | Chunks | Mean/meeting | Mean words | Median words | Mean turns |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for chunker in CHUNKERS:
        for size in result["sizes"]:
            row = result["chunking"][chunker][str(size)]
            lines.append(
                f"| {chunker} | {size} | {row['chunk_count']} | "
                f"{row['chunks_per_meeting']:.1f} | {row['mean_words']:.1f} | "
                f"{row['median_words']:.1f} | {row['mean_turns']:.1f} |"
            )

    lines += [
        "",
        "## Retrieval",
        "",
        "Values are meeting-macro averages. `dR` and `dF1` are paired differences from the 256-word condition with the same chunker, retriever, and evidence budget.",
        "",
        "| Chunker | Size | Retriever | Budget | Precision | Recall | F1 | Zero-hit | Clipping | MRR | dR | dF1 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, spec in result["conditions"].items():
        row = result["aggregate"][name]
        reference = result["aggregate"][
            condition_name(
                spec["chunker"], 256, spec["retriever"], spec["evidence_words"]
            )
        ]
        lines.append(
            f"| {spec['chunker']} | {spec['size']} | {spec['retriever']} | "
            f"{spec['evidence_words']} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | "
            f"{row['zero_hit']:.1%} | {row['clipped']:.1%} | "
            f"{row['mrr']:.3f} | {row['recall'] - reference['recall']:+.3f} | "
            f"{row['f1'] - reference['f1']:+.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Turn-packed sizes are soft limits because a long speaker turn remains intact.",
        "- Word-packed sizes are hard limits and may split a speaker turn.",
        "- MRR is size-sensitive because larger chunks are more likely to overlap a gold turn.",
        "- This sweep evaluates retrieval only; it does not measure answer-generation effects.",
        "",
    ]
    return "\n".join(lines)


def condition_name(chunker, size, retriever, budget) -> str:
    return f"{chunker}__s{size}__{retriever}__w{budget}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from meeting_qa_chunking.retrieval import (
        load_model,
        rank_chunks,
        rank_chunks_bm25,
        reciprocal_rank_fusion,
    )

    sizes = sorted(set(args.sizes))
    if 256 not in sizes or any(size <= 0 for size in sizes):
        raise ValueError("Sizes must be positive and include the 256-word reference")

    run = load_run_config(args.preset)
    spec = run.retrieval
    meeting_ids = run.meeting_ids()
    meetings = [
        load_meeting(run.data_dir / f"{meeting_id}.json")
        for meeting_id in meeting_ids
    ]
    model = load_model(spec.dense_model.name, spec.dense_model.revision)
    print(f"Embedding device: {model.device}", flush=True)

    chunks = {
        chunker: {
            size: {meeting.id: builder(meeting.turns, size) for meeting in meetings}
            for size in sizes
        }
        for chunker, builder in CHUNKERS.items()
    }
    conditions = {
        condition_name(chunker, size, retriever, budget): {
            "chunker": chunker,
            "size": size,
            "retriever": retriever,
            "evidence_words": budget,
        }
        for chunker in CHUNKERS
        for size in sizes
        for retriever in spec.retrievers
        for budget in spec.evidence_budgets
    }

    questions = []
    for meeting in meetings:
        for question_index, question in enumerate(meeting.questions):
            results = {}
            for chunker in CHUNKERS:
                for size in sizes:
                    units = chunks[chunker][size][meeting.id]
                    dense, _cache_hit = rank_chunks(
                        question.text,
                        units,
                        model,
                        spec.dense_model.name,
                        spec.dense_model.revision,
                    )
                    bm25 = rank_chunks_bm25(
                        question.text, units, spec.bm25_k1, spec.bm25_b
                    )
                    rankings = {
                        "dense": dense,
                        "bm25": bm25,
                        "hybrid": reciprocal_rank_fusion([dense, bm25], spec.rrf_k),
                    }
                    for retriever in spec.retrievers:
                        for budget in spec.evidence_budgets:
                            name = condition_name(
                                chunker, size, retriever, budget
                            )
                            results[name] = evaluate(
                                question,
                                meeting,
                                units,
                                rankings[retriever],
                                budget,
                            )
            questions.append(
                {
                    "meeting_id": meeting.id,
                    "question_index": question_index,
                    "question": question.text,
                    "results": results,
                }
            )
            print(f"Retrieval {meeting.id} question {question_index}", flush=True)

    result = {
        "preset": str(args.preset),
        "meeting_ids": meeting_ids,
        "question_count": len(questions),
        "sizes": sizes,
        "retrievers": list(spec.retrievers),
        "evidence_budgets": list(spec.evidence_budgets),
        "conditions": conditions,
        "chunking": {
            chunker: {
                str(size): chunk_summary(list(chunks[chunker][size].values()))
                for size in sizes
            }
            for chunker in CHUNKERS
        },
        "aggregate": aggregate(questions, meeting_ids, conditions),
        "questions": questions,
    }
    write_json(args.output, result)
    report = args.output.with_suffix(".md")
    report.write_text(make_report(result), encoding="utf-8")
    print(f"Sweep data: {args.output}")
    print(f"Sweep report: {report}")


if __name__ == "__main__":
    main()
