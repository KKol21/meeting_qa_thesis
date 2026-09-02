"""Step 3: find baseline chunks that overlap QMSum's gold evidence."""

import argparse
from pathlib import Path

from qmsum_data import load_meeting
from simple_chunking import chunk_by_word_budget


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--max-words", type=int, default=256)
    args = parser.parse_args()

    meeting = load_meeting(args.data_dir / f"{args.meeting}.json")
    question = meeting.questions[args.question_index]
    chunks = chunk_by_word_budget(meeting.turns, args.max_words)

    overlapping_chunks = [
        (index, chunk)
        for index, chunk in enumerate(chunks)
        if any(
            chunk.overlaps(start, end)
            for start, end in question.relevant_turn_ranges
        )
    ]

    print(f"Question: {question.text}")
    print(f"Gold evidence turns: {question.relevant_turn_ranges}")
    print(f"Overlapping chunks: {len(overlapping_chunks)} of {len(chunks)}\n")

    for index, chunk in overlapping_chunks:
        print(
            f"- Chunk {index}: turns {chunk.start_turn}-{chunk.end_turn}, "
            f"{chunk.word_count} words"
        )


if __name__ == "__main__":
    main()
