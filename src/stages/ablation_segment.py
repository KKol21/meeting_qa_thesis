"""Stage 1: create resumable Lumber segmentations from one preset."""

import argparse
from dataclasses import asdict
from pathlib import Path

from meeting_qa_chunking.artifacts import (
    EXPERIMENT_VERSION,
    make_provenance,
    read_json_object,
    write_json,
)
from meeting_qa_chunking.config import load_run_config
from meeting_qa_chunking.lumber import load_lumber_chunks, lumber_chunks
from meeting_qa_chunking.lumber_prompt import LUMBERCHUNKER_INSTRUCTIONS
from meeting_qa_chunking.qmsum import load_meeting
from meeting_qa_chunking.selection import select_meeting_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    args = parser.parse_args()

    from meeting_qa_chunking.local_model import LocalChatModel

    run = load_run_config(args.preset)
    spec = run.segmentation
    paths = select_meeting_paths(
        run.data_dir, len(run.meeting_ids()), 0, run.meeting_ids()
    )
    effective_config = {
        "experiment_version": EXPERIMENT_VERSION,
        "model": asdict(spec.model),
        "target_tokens": spec.target_tokens,
        "max_new_tokens": spec.max_new_tokens,
        "temperature": spec.temperature,
        "seed": spec.seed,
        "prompt": LUMBERCHUNKER_INSTRUCTIONS,
    }

    pending = []
    for path in paths:
        meeting = load_meeting(path)
        output_path = run.lumber_dir / path.name
        provenance = make_provenance(
            "segmentation", effective_config, {"meeting": path}, args.preset
        )
        if output_path.exists():
            try:
                saved = read_json_object(output_path)
            except ValueError:
                saved = {}
            if saved.get("provenance", {}).get("fingerprint") == provenance["fingerprint"]:
                load_lumber_chunks(output_path, meeting)
                print(f"Segmentation {meeting.id}: existing", flush=True)
                continue
        pending.append((meeting, output_path, provenance))

    if not pending:
        print("All segmentations already exist")
        return

    model = LocalChatModel(
        model_name=spec.model.name,
        revision=spec.model.revision,
        max_new_tokens=spec.max_new_tokens,
        seed=spec.seed,
        temperature=spec.temperature,
        cache_dir=Path(".cache/lumber"),
        prequantized=spec.model.prequantized,
    )
    for meeting, output_path, provenance in pending:
        responses: list[str] = []
        cache_hits: list[bool] = []
        attempts = 0

        def choose_and_record(prompt: str) -> str:
            nonlocal attempts
            attempts += 1
            response = model(prompt)
            source = "cache" if model.last_cache_hit else "model"
            print(f"{meeting.id} attempt {attempts} ({source})", flush=True)
            return response

        def record_decision(response: str) -> None:
            responses.append(response)
            cache_hits.append(model.last_cache_hit)

        calls_before = model.model_calls
        hits_before = model.cache_hits
        chunks = lumber_chunks(
            meeting.turns,
            choose_and_record,
            target_tokens=spec.target_tokens,
            record_decision=record_decision,
        )
        result = {
            "experiment_version": EXPERIMENT_VERSION,
            "provenance": provenance,
            "meeting_id": meeting.id,
            "turn_count": len(meeting.turns),
            "model_calls": model.model_calls - calls_before,
            "cache_hits": model.cache_hits - hits_before,
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
