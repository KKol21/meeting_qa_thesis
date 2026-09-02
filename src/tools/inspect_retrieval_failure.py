"""Inspect one failed retrieval without recomputing embeddings."""

import argparse
import json
from pathlib import Path

from meeting_qa_chunking.chunking import Chunk, chunk_by_word_budget, render_turn
from meeting_qa_chunking.lumber import load_lumber_chunks
from meeting_qa_chunking.qmsum import Turn, load_meeting


def preview(chunk: Chunk, length: int = 500) -> str:
    return chunk.text[:length].replace("\n", " ")


def preview_turns(turns: list[Turn], length: int = 500) -> str:
    text = "\n".join(render_turn(turn) for turn in turns)
    return text[:length].replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting", default="Bed002")
    parser.add_argument("--question-index", type=int, default=3)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/qmsum/data/ALL/val")
    )
    parser.add_argument(
        "--lumber-result", type=Path, default=Path("runs/lumber/qmsum/Bed002.json")
    )
    parser.add_argument(
        "--retrieval-result",
        type=Path,
        default=Path("runs/retrieval/qmsum/Bed002.json"),
    )
    args = parser.parse_args()

    meeting = load_meeting(args.data_dir / f"{args.meeting}.json")
    retrieval = json.loads(args.retrieval_result.read_text(encoding="utf-8"))
    question = meeting.questions[args.question_index]
    question_result = retrieval["questions"][args.question_index]

    fixed_chunks = chunk_by_word_budget(
        meeting.turns, retrieval["fixed"]["chunk_words"]
    )
    lumber_chunks = load_lumber_chunks(args.lumber_result, meeting)

    print(f"Question: {question.text}")
    print(f"Gold ranges: {question.relevant_turn_ranges}\n")
    for start, end in question.relevant_turn_ranges:
        print(
            f"Gold {start}-{end}: "
            f"{preview_turns(meeting.turns[start:end + 1])}...\n"
        )

    for name, chunks in (("Fixed", fixed_chunks), ("Lumber", lumber_chunks)):
        indices = question_result[name.lower()]["selected_chunk_indices"]
        for index in indices:
            chunk = chunks[index]
            print(
                f"{name} chunk {index}, turns {chunk.start_turn}-{chunk.end_turn}: "
                f"{preview(chunk)}...\n"
            )


if __name__ == "__main__":
    main()
