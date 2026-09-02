"""Step 1: load one QMSum meeting and inspect its contents."""

import argparse
from pathlib import Path

from qmsum_data import load_meeting


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--meeting", help="Meeting ID, for example Bed002")
    args = parser.parse_args()

    meeting_path = (
        args.data_dir / f"{args.meeting}.json"
        if args.meeting
        else next(iter(sorted(args.data_dir.glob("*.json"))), None)
    )
    if meeting_path is None or not meeting_path.exists():
        raise FileNotFoundError(f"No QMSum meeting found in {args.data_dir}")

    meeting = load_meeting(meeting_path)
    question = meeting.questions[0]

    print(f"Meeting: {meeting.id}")
    print(f"Transcript turns: {len(meeting.turns)}")
    print("\nFirst 3 turns:")
    for turn in meeting.turns[:3]:
        print(f"- [{turn.id}] {turn.speaker}: {turn.text}")

    print(f"\nQuestion: {question.text}")
    print(f"Reference answer: {question.reference_answer}")
    print(f"Relevant turn range(s): {question.relevant_turn_ranges}")


if __name__ == "__main__":
    main()
