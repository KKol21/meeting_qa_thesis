"""Step 7: simulate one LumberChunker boundary decision."""

import argparse
from pathlib import Path

from lumber_prompt import (
    build_window,
    estimate_tokens,
    parse_boundary,
    render_document,
)
from qmsum_data import load_meeting


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--target-tokens", type=int, default=550)
    parser.add_argument("--response", default="Answer: ID 0014")
    args = parser.parse_args()

    meeting = load_meeting(args.data_dir / f"{args.meeting}.json")
    first_window = build_window(meeting.turns, 0, args.target_tokens)
    boundary = parse_boundary(args.response, first_window)
    first_chunk = meeting.turns[:boundary]
    next_window = build_window(meeting.turns, boundary, args.target_tokens)

    print(f"First local window: turns {first_window[0].id}-{first_window[-1].id}")
    print(f"Simulated response: {args.response}")
    print(f"First semantic chunk: turns {first_chunk[0].id}-{first_chunk[-1].id}")
    print(f"Next local window: turns {next_window[0].id}-{next_window[-1].id}")
    print(
        "Next window estimated tokens: "
        f"{estimate_tokens(render_document(next_window))}"
    )


if __name__ == "__main__":
    main()
