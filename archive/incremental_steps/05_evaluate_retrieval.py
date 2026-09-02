"""Step 5: evaluate retrieved QMSum evidence under a fixed budget."""

import argparse
from pathlib import Path

from evidence_selection import score_evidence, select_evidence
from qmsum_data import load_meeting
from simple_chunking import chunk_by_word_budget
from simple_retrieval import load_model, rank_chunks


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--chunk-words", type=int, default=256)
    parser.add_argument("--evidence-words", type=int, default=512)
    args = parser.parse_args()

    meeting = load_meeting(args.data_dir / f"{args.meeting}.json")
    question = meeting.questions[args.question_index]
    chunks = chunk_by_word_budget(meeting.turns, args.chunk_words)
    ranking, cache_hit = rank_chunks(question.text, chunks, load_model())
    evidence = select_evidence(ranking, chunks, args.evidence_words)
    metrics = score_evidence(evidence, meeting, question)

    print(f"Question: {question.text}")
    print(f"Chunk embeddings: {'cache hit' if cache_hit else 'newly encoded'}")
    print(f"Selected chunks: {evidence.chunk_indices}")
    print(f"Retrieved words: {metrics.retrieved_words}")
    print(f"Relevant retrieved words: {metrics.relevant_retrieved_words}")
    print(f"Gold evidence words: {metrics.gold_words}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall: {metrics.recall:.3f}")


if __name__ == "__main__":
    main()
