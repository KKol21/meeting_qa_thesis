"""Step 2: split one QMSum meeting without semantic analysis."""

import argparse
from pathlib import Path

from qmsum_data import load_meeting
from simple_chunking import chunk_by_word_budget


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--max-words", type=int, default=256)
    args = parser.parse_args()

    meeting = load_meeting(args.data_dir / f"{args.meeting}.json")
    chunks = chunk_by_word_budget(meeting.turns, args.max_words)

    print(f"Meeting: {meeting.id}")
    print(f"Chunks: {len(chunks)} (target: {args.max_words} words each)")
    print("Complete speaker turns are never split.\n")

    for index, chunk in enumerate(chunks[:3]):
        preview = chunk.text[:250].replace("\n", " ")
        print(
            f"Chunk {index}: turns {chunk.start_turn}-{chunk.end_turn}, "
            f"{chunk.word_count} words"
        )
        print(f"{preview}...\n")


if __name__ == "__main__":
    main()
