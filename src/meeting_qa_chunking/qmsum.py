"""Small data structures and loader for one QMSum meeting."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Turn:
    id: int
    speaker: str
    text: str


@dataclass
class Question:
    text: str
    reference_answer: str
    relevant_turn_ranges: list[tuple[int, int]]


@dataclass
class Meeting:
    id: str
    turns: list[Turn]
    questions: list[Question]


def load_meeting(path: Path) -> Meeting:
    raw = json.loads(path.read_text(encoding="utf-8"))

    turns = [
        Turn(id=index, speaker=turn["speaker"], text=turn["content"])
        for index, turn in enumerate(raw["meeting_transcripts"])
    ]
    questions = [
        Question(
            text=question["query"],
            reference_answer=question["answer"],
            relevant_turn_ranges=[
                (int(start), int(end))
                for start, end in question["relevant_text_span"]
            ],
        )
        for question in raw["specific_query_list"]
    ]

    return Meeting(id=path.stem, turns=turns, questions=questions)
