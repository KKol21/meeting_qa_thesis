"""Step 6: print the first local LumberChunker prompt without calling a model."""

import argparse
from pathlib import Path

from lumber_prompt import build_prompt, build_window, estimate_tokens, render_document
from qmsum_data import load_meeting


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--start-turn", type=int, default=0)
    parser.add_argument("--target-tokens", type=int, default=550)
    args = parser.parse_args()

    meeting = load_meeting(args.data_dir / f"{args.meeting}.json")
    window = build_window(meeting.turns, args.start_turn, args.target_tokens)

    print(f"Meeting: {meeting.id}")
    print(f"Transcript turns: {len(meeting.turns)}")
    print(f"Local window: turns {window[0].id}-{window[-1].id}")
    print(f"Estimated document tokens: {estimate_tokens(render_document(window))}")
    print("\n--- PROMPT START ---\n")
    print(build_prompt(window))
    print("\n--- PROMPT END ---")


if __name__ == "__main__":
    main()
