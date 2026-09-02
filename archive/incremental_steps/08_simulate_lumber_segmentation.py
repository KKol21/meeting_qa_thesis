"""Step 8: verify the complete LumberChunker loop with a fake chooser."""

import argparse
from pathlib import Path
import re

from qmsum_data import load_meeting
from simple_lumber import lumber_chunks


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")
DOCUMENT_ID = re.compile(r"^ID (\d+):", re.MULTILINE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--target-tokens", type=int, default=550)
    args = parser.parse_args()

    meeting = load_meeting(args.data_dir / f"{args.meeting}.json")
    prompt_sizes: list[int] = []

    def fake_midpoint_chooser(prompt: str) -> str:
        """Return the middle ID; this tests control flow, not semantics."""

        ids = [int(value) for value in DOCUMENT_ID.findall(prompt)]
        prompt_sizes.append(len(ids))
        return f"Answer: ID {ids[len(ids) // 2]:04d}"

    chunks = lumber_chunks(
        meeting.turns,
        fake_midpoint_chooser,
        args.target_tokens,
    )

    covered_turns = [turn.id for chunk in chunks for turn in chunk.turns]
    assert covered_turns == list(range(len(meeting.turns)))
    assert max(prompt_sizes, default=0) < len(meeting.turns)

    print("Simulation only: midpoint boundaries are not semantic results.")
    print(f"Meeting turns: {len(meeting.turns)}")
    print(f"Produced chunks: {len(chunks)}")
    print(f"Fake model calls: {len(prompt_sizes)}")
    print(f"Largest local prompt: {max(prompt_sizes, default=0)} turns")
    print("All turns covered exactly once: yes\n")

    for index, chunk in enumerate(chunks[:5]):
        print(f"Chunk {index}: turns {chunk.start_turn}-{chunk.end_turn}")
    if len(chunks) > 5:
        last = chunks[-1]
        print("...")
        print(f"Chunk {len(chunks) - 1}: turns {last.start_turn}-{last.end_turn}")


if __name__ == "__main__":
    main()
