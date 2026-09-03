"""Inspect one saved retrieval condition without loading a model."""

import argparse
import json
from pathlib import Path

from meeting_qa_chunking.evidence import (
    reconstruct_evidence,
    render_evidence,
    render_gold_evidence,
)
from meeting_qa_chunking.evidence_preparation import build_chunk_sets
from meeting_qa_chunking.config import load_run_config
from meeting_qa_chunking.qmsum import load_meeting


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--meeting")
    parser.add_argument("--question-index", type=int, default=3)
    parser.add_argument("--condition")
    args = parser.parse_args()

    run = load_run_config(args.preset)
    meeting_id = args.meeting or run.meeting_ids()[0]
    meeting = load_meeting(run.data_dir / f"{meeting_id}.json")
    retrieval = json.loads(
        (run.retrieval_dir / f"{meeting_id}.json").read_text(
            encoding="utf-8"
        )
    )
    question_result = retrieval["questions"][args.question_index]
    condition = args.condition or next(iter(retrieval["configurations"]))
    condition_spec = retrieval["configurations"][condition]
    chunker = condition_spec["chunker"]
    chunks = build_chunk_sets(
        meeting,
        run.lumber_dir / f"{meeting.id}.json" if chunker == "lumber" else None,
        retrieval["chunking"]["turn_packed_max_words"],
        retrieval["chunking"]["word_packed_max_words"],
        [chunker],
    )[chunker]
    selected = reconstruct_evidence(
        question_result["results"][condition],
        chunks,
        condition_spec["evidence_words"],
    )
    question = meeting.questions[args.question_index]
    gold, _turn_ids = render_gold_evidence(question, meeting)

    print(f"Question: {question.text}")
    print(f"Condition: {condition}")
    print(f"Selected chunks: {selected.chunk_indices}")
    print(f"Gold ranges: {question.relevant_turn_ranges}\n")
    print("Gold evidence:\n" + gold)
    print(
        "\nRetrieved evidence:\n"
        + render_evidence(selected, meeting, retrieval["evidence_order"])
    )


if __name__ == "__main__":
    main()
