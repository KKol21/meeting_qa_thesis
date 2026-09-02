"""Stage 1: create resumable Lumber segmentations for a meeting set."""

import argparse
from pathlib import Path

from meeting_qa_chunking.boundary_model import LocalBoundaryChooser
from meeting_qa_chunking.config import LUMBER_MODEL
from meeting_qa_chunking.experiment import EXPERIMENT_VERSION, select_meeting_paths, write_json
from meeting_qa_chunking.lumber import load_lumber_chunks, lumber_chunks
from meeting_qa_chunking.qmsum import load_meeting


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")
DEFAULT_OUTPUT_DIR = Path("runs/lumber/qmsum")
DEFAULT_MODEL = LUMBER_MODEL.name
DEFAULT_REVISION = LUMBER_MODEL.revision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meetings", nargs="+")
    parser.add_argument("--target-tokens", type=int, default=550)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    paths = select_meeting_paths(
        args.data_dir, args.count, args.seed, args.meetings
    )
    # A saved meeting is the resume unit; validate it before skipping work.
    pending = []
    for path in paths:
        meeting = load_meeting(path)
        output_path = args.output_dir / f"{meeting.id}.json"
        if output_path.exists():
            load_lumber_chunks(output_path, meeting)
            print(f"Segmentation {meeting.id}: existing", flush=True)
        else:
            pending.append((meeting, output_path))

    if not pending:
        print("All segmentations already exist")
        return

    # Delay the expensive model load until at least one meeting is missing.
    chooser = LocalBoundaryChooser(
        model_name=DEFAULT_MODEL,
        revision=DEFAULT_REVISION,
        max_new_tokens=args.max_new_tokens,
    )
    for meeting, output_path in pending:
        responses: list[str] = []
        cache_hits: list[bool] = []
        attempts = 0

        def choose_and_record(prompt: str) -> str:
            nonlocal attempts
            attempts += 1
            response = chooser(prompt)
            source = "cache" if chooser.last_cache_hit else "model"
            print(
                f"{meeting.id} attempt {attempts} ({source})",
                flush=True,
            )
            return response

        def record_decision(response: str) -> None:
            responses.append(response)
            cache_hits.append(chooser.last_cache_hit)

        calls_before = chooser.model_calls
        hits_before = chooser.cache_hits
        chunks = lumber_chunks(
            meeting.turns,
            choose_and_record,
            target_tokens=args.target_tokens,
            record_decision=record_decision,
        )
        result = {
            "experiment_version": EXPERIMENT_VERSION,
            "meeting_id": meeting.id,
            "turn_count": len(meeting.turns),
            "model": DEFAULT_MODEL,
            "revision": DEFAULT_REVISION,
            "target_tokens": args.target_tokens,
            "max_new_tokens": args.max_new_tokens,
            "model_calls": chooser.model_calls - calls_before,
            "cache_hits": chooser.cache_hits - hits_before,
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
                for chunk, response, cache_hit in zip(
                    chunks, responses, cache_hits
                )
            ],
        }
        write_json(output_path, result)
        print(f"Segmentation {meeting.id}: {len(chunks)} chunks saved", flush=True)


if __name__ == "__main__":
    main()
