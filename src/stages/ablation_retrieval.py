"""Stage 2: evaluate the retrieval grid on a fixed QMSum meeting set."""

import argparse
import json
from pathlib import Path
from statistics import mean

from meeting_qa_chunking.config import ConditionSpec, retrieval_conditions
from meeting_qa_chunking.evidence import (
    first_relevant_chunk_rank,
    score_evidence,
    select_evidence,
)
from meeting_qa_chunking.experiment import (
    EXPERIMENT_VERSION,
    select_meeting_paths,
    write_json,
)
from meeting_qa_chunking.pipeline import build_chunk_sets
from meeting_qa_chunking.qmsum import load_meeting
from meeting_qa_chunking.retrieval import (
    BM25_B,
    BM25_K1,
    MODEL_NAME,
    MODEL_REVISION,
    RRF_K,
    load_model,
    rank_chunks,
    rank_chunks_bm25,
    reciprocal_rank_fusion,
)


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")
DEFAULT_LUMBER_DIR = Path("runs/lumber/qmsum")
DEFAULT_OUTPUT_DIR = Path("runs/ablations/full/retrieval")


def configuration_name(chunker: str, retriever: str, words: int) -> str:
    return ConditionSpec(chunker, retriever, words).name


def configurations(evidence_budgets: list[int]) -> dict[str, dict[str, object]]:
    return {
        condition.name: condition.to_dict()
        for condition in retrieval_conditions(evidence_budgets)
    }


def summarize(output_dir: Path, meeting_ids: list[str]) -> dict[str, object]:
    results = [
        json.loads((output_dir / f"{meeting_id}.json").read_text(encoding="utf-8"))
        for meeting_id in meeting_ids
    ]
    condition_names = list(results[0]["configurations"])
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": "retrieval",
        "meeting_ids": meeting_ids,
        "meeting_count": len(meeting_ids),
        "question_count": sum(result["question_count"] for result in results),
        "configurations": results[0]["configurations"],
        "macro_average": {
            condition: {
                metric: mean(
                    question["results"][condition][metric]
                    for result in results
                    for question in result["questions"]
                )
                for metric in ("precision", "recall", "reciprocal_rank")
            }
            for condition in condition_names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--lumber-dir", type=Path, default=DEFAULT_LUMBER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meetings", nargs="+")
    parser.add_argument("--fixed-chunk-words", type=int, default=256)
    parser.add_argument("--evidence-budgets", type=int, nargs="+", default=[512, 1024])
    args = parser.parse_args()

    if any(words <= 0 for words in args.evidence_budgets):
        raise ValueError("Evidence budgets must be positive")
    paths = select_meeting_paths(
        args.data_dir, args.count, args.seed, args.meetings
    )
    config = configurations(args.evidence_budgets)
    # Results are resumable per meeting, but only if the grid is unchanged.
    pending = []
    for path in paths:
        output_path = args.output_dir / path.name
        if output_path.exists():
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                saved.get("experiment_version") == EXPERIMENT_VERSION
                and saved.get("configurations") == config
            ):
                print(f"Retrieval {path.stem}: existing", flush=True)
                continue
        pending.append((path, output_path))

    model = load_model() if pending else None
    if model is not None:
        print(f"Embedding device: {model.device}", flush=True)

    for path, output_path in pending:
        meeting = load_meeting(path)
        chunk_sets = build_chunk_sets(
            meeting,
            args.lumber_dir / f"{meeting.id}.json",
            args.fixed_chunk_words,
        )
        questions = []
        dense_cache_hits = {}
        for question_index, question in enumerate(meeting.questions):
            question_results = {}
            for chunker, chunks in chunk_sets.items():
                dense, cache_hit = rank_chunks(question.text, chunks, model)
                dense_cache_hits.setdefault(chunker, cache_hit)
                bm25 = rank_chunks_bm25(question.text, chunks)
                # Dense and BM25 are computed once, then reused across budgets.
                rankings = {
                    "dense": dense,
                    "bm25": bm25,
                    "hybrid": reciprocal_rank_fusion([dense, bm25]),
                }
                for retriever, ranking in rankings.items():
                    first_gold_rank = first_relevant_chunk_rank(
                        ranking, chunks, question
                    )
                    for words in args.evidence_budgets:
                        evidence = select_evidence(ranking, chunks, words)
                        metrics = score_evidence(evidence, meeting, question)
                        name = configuration_name(chunker, retriever, words)
                        question_results[name] = {
                            "precision": metrics.precision,
                            "recall": metrics.recall,
                            "first_gold_rank": first_gold_rank,
                            "reciprocal_rank": (
                                1 / first_gold_rank if first_gold_rank else 0.0
                            ),
                            "retrieved_words": metrics.retrieved_words,
                            "relevant_retrieved_words": (
                                metrics.relevant_retrieved_words
                            ),
                            "gold_words": metrics.gold_words,
                            "selected_chunk_indices": evidence.chunk_indices,
                        }
            questions.append(
                {
                    "question_index": question_index,
                    "question": question.text,
                    "reference_answer": question.reference_answer,
                    "results": question_results,
                }
            )

        result = {
            "experiment_version": EXPERIMENT_VERSION,
            "meeting_id": meeting.id,
            "question_count": len(questions),
            "configurations": config,
            "fixed_chunk_words": args.fixed_chunk_words,
            "retrievers": {
                "dense": {"model": MODEL_NAME, "revision": MODEL_REVISION},
                "bm25": {"k1": BM25_K1, "b": BM25_B},
                "hybrid": {"method": "reciprocal_rank_fusion", "k": RRF_K},
            },
            "dense_embeddings_initially_cached": dense_cache_hits,
            "chunk_counts": {
                name: len(chunks) for name, chunks in chunk_sets.items()
            },
            "questions": questions,
        }
        write_json(output_path, result)
        print(f"Retrieval {meeting.id}: saved", flush=True)

    meeting_ids = [path.stem for path in paths]
    summary = summarize(args.output_dir, meeting_ids)
    write_json(args.output_dir / "summary.json", summary)
    print(f"Retrieval summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
