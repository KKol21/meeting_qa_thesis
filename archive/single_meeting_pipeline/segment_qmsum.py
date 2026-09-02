"""Segment one complete QMSum meeting and save the result."""

import argparse
import json
from pathlib import Path

from meeting_qa_chunking.boundary_model import LocalBoundaryChooser
from meeting_qa_chunking.lumber import lumber_chunks
from meeting_qa_chunking.qmsum import load_meeting


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")
DEFAULT_OUTPUT_DIR = Path("runs/lumber/qmsum")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--target-tokens", type=int, default=550)
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
    attempts = 0

    def choose_and_record(prompt: str) -> str:
        nonlocal attempts
        attempts += 1
        response = chooser(prompt)
        source = "cache" if chooser.last_cache_hit else "model"
        first_line = response.splitlines()[0] if response else ""
        print(f"Attempt {attempts} ({source}): {first_line!r}", flush=True)
        return response

    def record_decision(response: str) -> None:
        responses.append(response)
        cache_hits.append(chooser.last_cache_hit)

    chunks = lumber_chunks(
        meeting.turns,
        choose_and_record,
        target_tokens=args.target_tokens,
        record_decision=record_decision,
    )
    covered_turns = [turn.id for chunk in chunks for turn in chunk.turns]
    if covered_turns != list(range(len(meeting.turns))):
        raise RuntimeError("Chunks do not cover every turn exactly once")

    result = {
        "meeting_id": meeting.id,
        "turn_count": len(meeting.turns),
        "model": args.model,
        "revision": revision,
        "target_tokens": args.target_tokens,
        "max_new_tokens": args.max_new_tokens,
        "model_calls": chooser.model_calls,
        "cache_hits": chooser.cache_hits,
        "chunks": [
            {
                "index": chunk.index,
                "start_turn": chunk.start_turn,
                "end_turn": chunk.end_turn,
                "word_count": chunk.word_count,
            }
            for chunk in chunks
        ],
        "decisions": [
            {
                "boundary_turn": chunk.end_turn + 1,
                "raw_response": response,
                "cache_hit": cache_hit,
            }
            for chunk, response, cache_hit in zip(chunks, responses, cache_hits)
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{meeting.id}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Chunks: {len(chunks)}")
    print(f"All {len(meeting.turns)} turns segmented: yes")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
