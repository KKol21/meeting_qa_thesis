"""Compare fixed and Lumber chunks on one QMSum meeting."""

import argparse
import json
from pathlib import Path
from statistics import mean

from sentence_transformers import SentenceTransformer

from meeting_qa_chunking.chunking import Chunk, chunk_by_word_budget
from meeting_qa_chunking.evidence import first_relevant_chunk_rank, score_evidence, select_evidence
from meeting_qa_chunking.lumber import load_lumber_chunks
from meeting_qa_chunking.qmsum import Meeting, Question, load_meeting
from meeting_qa_chunking.retrieval import MODEL_NAME, MODEL_REVISION, load_model, rank_chunks


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")
DEFAULT_LUMBER_RESULT = Path("runs/lumber/qmsum/Bed002.json")
DEFAULT_OUTPUT = Path("runs/retrieval/qmsum/Bed002.json")


def evaluate(
    question: Question,
    chunks: list[Chunk],
    meeting: Meeting,
    model: SentenceTransformer,
    evidence_words: int,
) -> tuple[dict[str, object], bool]:
    ranking, cache_hit = rank_chunks(question.text, chunks, model)
    evidence = select_evidence(ranking, chunks, evidence_words)
    metrics = score_evidence(evidence, meeting, question)
    first_gold_rank = first_relevant_chunk_rank(ranking, chunks, question)
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "first_gold_rank": first_gold_rank,
        "reciprocal_rank": 1 / first_gold_rank if first_gold_rank else 0.0,
        "retrieved_words": metrics.retrieved_words,
        "relevant_retrieved_words": metrics.relevant_retrieved_words,
        "gold_words": metrics.gold_words,
        "selected_chunk_indices": evidence.chunk_indices,
    }, cache_hit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--lumber-result", type=Path, default=DEFAULT_LUMBER_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixed-chunk-words", type=int, default=256)
    parser.add_argument("--evidence-words", type=int, default=512)
    args = parser.parse_args()

    meeting = load_meeting(args.data_dir / f"{args.meeting}.json")
    fixed_chunks = chunk_by_word_budget(meeting.turns, args.fixed_chunk_words)
    lumber_chunks = load_lumber_chunks(args.lumber_result, meeting)
    model = load_model()
    print(f"Embedding device: {model.device}", flush=True)

    questions = []
    fixed_cache_hits = []
    lumber_cache_hits = []
    for index, question in enumerate(meeting.questions):
        fixed, fixed_cache_hit = evaluate(
            question, fixed_chunks, meeting, model, args.evidence_words
        )
        lumber, lumber_cache_hit = evaluate(
            question, lumber_chunks, meeting, model, args.evidence_words
        )
        fixed_cache_hits.append(fixed_cache_hit)
        lumber_cache_hits.append(lumber_cache_hit)
        questions.append(
            {
                "question_index": index,
                "question": question.text,
                "fixed": fixed,
                "lumber": lumber,
            }
        )

    averages = {
        method: {
            metric: mean(item[method][metric] for item in questions)
            for metric in ("precision", "recall", "reciprocal_rank")
        }
        for method in ("fixed", "lumber")
    }
    result = {
        "meeting_id": meeting.id,
        "question_count": len(questions),
        "retriever": {"model": MODEL_NAME, "revision": MODEL_REVISION},
        "evidence_words": args.evidence_words,
        "fixed": {
            "chunk_words": args.fixed_chunk_words,
            "chunk_count": len(fixed_chunks),
            "embeddings_initially_cached": fixed_cache_hits[0],
        },
        "lumber": {
            "result": str(args.lumber_result),
            "chunk_count": len(lumber_chunks),
            "embeddings_initially_cached": lumber_cache_hits[0],
        },
        "macro_average": averages,
        "questions": questions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Question  Fixed P  Fixed R  F rank  Lumber P  Lumber R  L rank")
    for item in questions:
        fixed_rank = item["fixed"]["first_gold_rank"] or "-"
        lumber_rank = item["lumber"]["first_gold_rank"] or "-"
        print(
            f"{item['question_index']:>8}  "
            f"{item['fixed']['precision']:.3f}    "
            f"{item['fixed']['recall']:.3f}    "
            f"{fixed_rank!s:>6}  "
            f"{item['lumber']['precision']:.3f}     "
            f"{item['lumber']['recall']:.3f}  "
            f"{lumber_rank!s:>6}"
        )
    print(
        "Macro avg  "
        f"{averages['fixed']['precision']:.3f}    "
        f"{averages['fixed']['recall']:.3f}    "
        "        "
        f"{averages['lumber']['precision']:.3f}     "
        f"{averages['lumber']['recall']:.3f}"
    )
    print(
        f"MRR: fixed={averages['fixed']['reciprocal_rank']:.3f}, "
        f"lumber={averages['lumber']['reciprocal_rank']:.3f}"
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
