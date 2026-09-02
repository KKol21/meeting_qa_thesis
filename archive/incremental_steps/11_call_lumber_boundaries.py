"""Step 11: make a small sequence of real LumberChunker boundary calls."""

import argparse
from pathlib import Path

from local_boundary_model import LocalBoundaryChooser
from qmsum_data import load_meeting
from simple_lumber import lumber_chunks


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--target-tokens", type=int, default=550)
    parser.add_argument("--max-boundaries", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    revision = args.revision
    if args.model == DEFAULT_MODEL and revision is None:
        revision = DEFAULT_REVISION

    meeting = load_meeting(args.data_dir / f"{args.meeting}.json")
    chooser = LocalBoundaryChooser(
        model_name=args.model,
        revision=revision,
        max_new_tokens=args.max_new_tokens,
    )
    responses: list[str] = []
    cache_hits: list[bool] = []

    def choose_and_record(prompt: str) -> str:
        response = chooser(prompt)
        responses.append(response)
        cache_hits.append(chooser.last_cache_hit)
        return response

    chunks = lumber_chunks(
        meeting.turns,
        choose_and_record,
        target_tokens=args.target_tokens,
        max_boundaries=args.max_boundaries,
    )

    print(f"Model: {args.model}")
    print(f"Revision: {revision or 'unpinned'}")
    print(f"Device: {chooser.device}")
    for index, (chunk, response, cache_hit) in enumerate(
        zip(chunks, responses, cache_hits),
        start=1,
    ):
        source = "cache" if cache_hit else "model"
        print(
            f"Boundary {index}: {response!r} -> "
            f"chunk {chunk.start_turn}-{chunk.end_turn} ({source})"
        )

    next_turn = chunks[-1].end_turn + 1 if chunks else 0
    if next_turn < len(meeting.turns):
        print(f"Next unprocessed turn: {next_turn}")
    else:
        print("Meeting fully segmented")
    print(f"Model calls: {chooser.model_calls}")
    print(f"Cache hits: {chooser.cache_hits}")


if __name__ == "__main__":
    main()
