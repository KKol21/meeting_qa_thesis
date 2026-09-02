"""Step 9: make one real local-model LumberChunker boundary call."""

import argparse
from pathlib import Path

from local_boundary_model import LocalBoundaryChooser
from lumber_prompt import build_prompt, build_window, parse_boundary
from qmsum_data import load_meeting


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--target-tokens", type=int, default=550)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    revision = args.revision
    if args.model == DEFAULT_MODEL and revision is None:
        revision = DEFAULT_REVISION

    meeting = load_meeting(args.data_dir / f"{args.meeting}.json")
    window = build_window(meeting.turns, 0, args.target_tokens)
    chooser = LocalBoundaryChooser(
        model_name=args.model,
        revision=revision,
        max_new_tokens=args.max_new_tokens,
    )
    response = chooser(build_prompt(window))

    print(f"Model: {args.model}")
    print(f"Revision: {revision or 'unpinned'}")
    print(f"Device: {chooser.device}")
    print(f"Local window: turns {window[0].id}-{window[-1].id}")
    print(f"Raw response: {response!r}")
    print(f"Response cache: {'hit' if chooser.last_cache_hit else 'new'}")

    try:
        boundary = parse_boundary(response, window)
    except ValueError as error:
        print(f"Valid boundary: no ({error})")
        return

    print(f"Valid boundary: yes, turn {boundary}")
    print(f"First semantic chunk: turns 0-{boundary - 1}")
    print(f"Next local window starts at: turn {boundary}")


if __name__ == "__main__":
    main()
