"""Stage 2: evaluate the retrieval grid defined by one preset."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean

from meeting_qa_chunking.artifacts import (
    EXPERIMENT_VERSION,
    make_provenance,
    questions_complete,
    read_retrieval,
    sha256_file,
    write_json,
)
from meeting_qa_chunking.config import (
    ConditionSpec,
    load_run_config,
    retrieval_conditions,
)
from meeting_qa_chunking.evidence import (
    first_relevant_chunk_rank,
    score_evidence,
    select_evidence,
)
from meeting_qa_chunking.evidence_preparation import build_chunk_sets
from meeting_qa_chunking.qmsum import load_meeting
from meeting_qa_chunking.selection import select_meeting_paths


METRICS = ("precision", "recall", "first_overlap_reciprocal_rank")


def configurations(run) -> dict[str, dict[str, object]]:
    return {
        condition.name: condition.to_dict()
        for condition in retrieval_conditions(
            run.retrieval.evidence_budgets,
            run.retrieval.chunkers,
            run.retrieval.retrievers,
        )
    }


def summarize(output_dir: Path, meeting_ids: list[str]) -> dict[str, object]:
    results = [
        json.loads((output_dir / f"{meeting_id}.json").read_text(encoding="utf-8"))
        for meeting_id in meeting_ids
    ]
    names = list(results[0]["configurations"])
    per_meeting = {
        result["meeting_id"]: {
            condition: {
                metric: mean(
                    question["results"][condition][metric]
                    for question in result["questions"]
                )
                for metric in METRICS
            }
            for condition in names
        }
        for result in results
    }
    meeting_average = {
        condition: {
            metric: mean(
                per_meeting[meeting_id][condition][metric]
                for meeting_id in meeting_ids
            )
            for metric in METRICS
        }
        for condition in names
    }

    paired = {}
    for name, condition in results[0]["configurations"].items():
        if condition["chunker"] != "lumber":
            continue
        suffix = f"{condition['retriever']}__w{condition['evidence_words']}"
        for baseline in ("turn_packed", "word_packed"):
            baseline_name = f"{baseline}__{suffix}"
            if baseline_name not in names:
                continue
            comparison = f"lumber_minus_{baseline}__{suffix}"
            paired[comparison] = {
                metric: mean(
                    per_meeting[meeting_id][name][metric]
                    - per_meeting[meeting_id][baseline_name][metric]
                    for meeting_id in meeting_ids
                )
                for metric in ("precision", "recall")
            }

    return {
        "experiment_version": EXPERIMENT_VERSION,
        "stage": "retrieval",
        "meeting_ids": meeting_ids,
        "meeting_count": len(meeting_ids),
        "question_count": sum(result["question_count"] for result in results),
        "configurations": results[0]["configurations"],
        "question_average": {
            condition: {
                metric: mean(
                    question["results"][condition][metric]
                    for result in results
                    for question in result["questions"]
                )
                for metric in METRICS
            }
            for condition in names
        },
        "per_meeting": per_meeting,
        "meeting_average": meeting_average,
        "paired_meeting_average": paired,
        "artifact_hashes": {
            meeting_id: sha256_file(output_dir / f"{meeting_id}.json")
            for meeting_id in meeting_ids
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
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
    paths = select_meeting_paths(run.data_dir, len(meeting_ids), 0, meeting_ids)
    condition_config = configurations(run)
    uses_dense = any(name in ("dense", "hybrid") for name in spec.retrievers)
    uses_bm25 = any(name in ("bm25", "hybrid") for name in spec.retrievers)
    effective_config = {
        "experiment_version": EXPERIMENT_VERSION,
        "configurations": condition_config,
        "chunkers": spec.chunkers,
        "retrievers": spec.retrievers,
        "turn_packed_max_words": spec.turn_packed_max_words,
        "word_packed_max_words": spec.word_packed_max_words,
        "evidence_order": spec.evidence_order,
    }
    if uses_dense:
        effective_config["dense_model"] = asdict(spec.dense_model)
    if uses_bm25:
        effective_config["bm25"] = {"k1": spec.bm25_k1, "b": spec.bm25_b}
    if "hybrid" in spec.retrievers:
        effective_config["rrf_k"] = spec.rrf_k

    pending = []
    for path in paths:
        meeting = load_meeting(path)
        lumber_path = run.lumber_dir / path.name
        uses_lumber = "lumber" in spec.chunkers
        if uses_lumber and not lumber_path.exists():
            raise FileNotFoundError(lumber_path)
        output_path = run.retrieval_dir / path.name
        inputs = {"meeting": path}
        if uses_lumber:
            inputs["segmentation"] = lumber_path
        provenance = make_provenance(
            "retrieval",
            effective_config,
            inputs,
            args.preset,
        )
        if output_path.exists():
            try:
                saved = read_retrieval(output_path)
            except ValueError:
                saved = {}
            if (
                saved.get("provenance", {}).get("fingerprint")
                == provenance["fingerprint"]
                and saved.get("configurations") == condition_config
                and questions_complete(
                    saved,
                    meeting.id,
                    [question.text for question in meeting.questions],
                    set(condition_config),
                )
            ):
                print(f"Retrieval {path.stem}: existing", flush=True)
                continue
        pending.append((meeting, output_path, provenance))

    model = None
    if pending and uses_dense:
        model = load_model(spec.dense_model.name, spec.dense_model.revision)
        print(f"Embedding device: {model.device}", flush=True)

    for meeting, output_path, provenance in pending:
        chunk_sets = build_chunk_sets(
            meeting,
            run.lumber_dir / f"{meeting.id}.json",
            spec.turn_packed_max_words,
            spec.word_packed_max_words,
            spec.chunkers,
        )
        questions = []
        dense_cache_hits = {}
        for question_index, question in enumerate(meeting.questions):
            question_results = {}
            for chunker in spec.chunkers:
                chunks = chunk_sets[chunker]
                all_rankings = {}
                if uses_dense:
                    dense, cache_hit = rank_chunks(
                        question.text,
                        chunks,
                        model,
                        spec.dense_model.name,
                        spec.dense_model.revision,
                    )
                    dense_cache_hits.setdefault(chunker, cache_hit)
                    all_rankings["dense"] = dense
                if uses_bm25:
                    all_rankings["bm25"] = rank_chunks_bm25(
                        question.text, chunks, spec.bm25_k1, spec.bm25_b
                    )
                if "hybrid" in spec.retrievers:
                    all_rankings["hybrid"] = reciprocal_rank_fusion(
                        [all_rankings["dense"], all_rankings["bm25"]],
                        spec.rrf_k,
                    )
                for retriever in spec.retrievers:
                    ranking = all_rankings[retriever]
                    first_rank = first_relevant_chunk_rank(ranking, chunks, question)
                    for words in spec.evidence_budgets:
                        evidence = select_evidence(ranking, chunks, words)
                        metrics = score_evidence(evidence, meeting, question)
                        name = ConditionSpec(chunker, retriever, words).name
                        question_results[name] = {
                            "precision": metrics.precision,
                            "recall": metrics.recall,
                            "first_overlap_rank": first_rank,
                            "first_overlap_reciprocal_rank": (
                                1 / first_rank if first_rank else 0.0
                            ),
                            "retrieved_words": metrics.retrieved_words,
                            "relevant_retrieved_words": metrics.relevant_retrieved_words,
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
            "provenance": provenance,
            "meeting_id": meeting.id,
            "question_count": len(questions),
            "configurations": condition_config,
            "chunking": {
                "turn_packed_max_words": spec.turn_packed_max_words,
                "word_packed_max_words": spec.word_packed_max_words,
            },
            "evidence_order": spec.evidence_order,
            "retrievers": {
                name: (
                    asdict(spec.dense_model)
                    if name == "dense"
                    else {"k1": spec.bm25_k1, "b": spec.bm25_b}
                    if name == "bm25"
                    else {
                        "method": "reciprocal_rank_fusion",
                        "k": spec.rrf_k,
                    }
                )
                for name in spec.retrievers
            },
            "dense_embeddings_initially_cached": dense_cache_hits,
            "chunk_counts": {
                name: len(chunks) for name, chunks in chunk_sets.items()
            },
            "questions": questions,
        }
        write_json(output_path, result)
        print(f"Retrieval {meeting.id}: saved", flush=True)

    write_json(run.retrieval_dir / "summary.json", summarize(run.retrieval_dir, meeting_ids))
    print(f"Retrieval summary: {run.retrieval_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
