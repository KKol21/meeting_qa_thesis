"""Step 4: retrieve chunks for one QMSum question."""

import argparse
from pathlib import Path

from qmsum_data import load_meeting
from simple_chunking import chunk_by_word_budget
from simple_retrieval import load_model, rank_chunks


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--max-words", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    meeting = load_meeting(args.data_dir / f"{args.meeting}.json")
    question = meeting.questions[args.question_index]
    chunks = chunk_by_word_budget(meeting.turns, args.max_words)
    ranking, cache_hit = rank_chunks(question.text, chunks, load_model())

    print(f"Question: {question.text}")
    print(f"Chunk embeddings: {'cache hit' if cache_hit else 'newly encoded'}")
    print(f"\nTop {args.top_k} chunks:\n")

    for rank, (chunk_index, score) in enumerate(ranking[: args.top_k], start=1):
        chunk = chunks[chunk_index]
        overlaps_gold = any(
            chunk.overlaps(start, end)
            for start, end in question.relevant_turn_ranges
        )
        label = "gold overlap" if overlaps_gold else "not gold"
        preview = chunk.text[:180].replace("\n", " ")

        print(
            f"{rank}. Chunk {chunk_index} | score {score:.3f} | "
            f"turns {chunk.start_turn}-{chunk.end_turn} | {label}"
        )
        print(f"   {preview}...\n")


if __name__ == "__main__":
    main()
