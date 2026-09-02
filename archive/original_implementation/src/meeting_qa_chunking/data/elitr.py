from __future__ import annotations

import json
from pathlib import Path
import re

from ..schema import DatasetSplit, Meeting, Query, Turn


_SPEAKER_LINE = re.compile(r"^\(([^)]+)\)\s*(.*)$")
_QUESTION_TYPES = {"who", "what", "howmany", "when"}
_ANSWER_POSITIONS = {"B", "M", "E", "S"}


def parse_transcript(path: str | Path) -> tuple[Turn, ...]:
    """Parse the ELITR format: ``(SPEAKER) text`` or same-speaker ``text``."""

    current_speaker = "UNKNOWN"
    turns: list[Turn] = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            text = line.strip()
            if not text:
                continue
            match = _SPEAKER_LINE.match(text)
            if match:
                current_speaker, text = match.groups()
                text = text.strip()
            if text:
                turns.append(Turn(len(turns), current_speaker, text))
    return tuple(turns)


def _transcript_index(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}

    def add(key: str, path: Path) -> None:
        if key in index and index[key] != path:
            raise ValueError(f"Ambiguous transcript id {key!r}")
        index[key] = path

    for path in directory.rglob("transcript_MAN*.txt"):
        add(path.parent.name, path)
    for path in directory.glob("*.txt"):
        add(path.stem, path)
    return index


def load_elitr(benchmark_path: str | Path, transcript_dir: str | Path) -> DatasetSplit:
    benchmark_path = Path(benchmark_path)
    with benchmark_path.open(encoding="utf-8") as source:
        raw = json.load(source)

    split = str(raw["split"])
    transcripts = _transcript_index(Path(transcript_dir))
    meetings: list[Meeting] = []
    queries: list[Query] = []

    for item in raw["meetings"]:
        meeting_id = str(item["id"])
        try:
            transcript_path = transcripts[meeting_id]
        except KeyError as error:
            raise FileNotFoundError(f"No transcript found for {meeting_id!r}") from error

        meeting = Meeting(
            dataset="elitr",
            split=split,
            id=meeting_id,
            turns=parse_transcript(transcript_path),
            metadata={"transcript_path": str(transcript_path)},
        )
        meetings.append(meeting)

        for question in item["questions"]:
            question_type = str(question["question-type"])
            answer_position = str(question["answer-position"])
            if question_type not in _QUESTION_TYPES:
                raise ValueError(f"Unexpected ELITR question type {question_type!r}")
            if answer_position not in _ANSWER_POSITIONS:
                raise ValueError(f"Unexpected ELITR answer position {answer_position!r}")
            queries.append(
                Query(
                    dataset="elitr",
                    split=split,
                    id=f"{meeting_id}:q{question['id']}",
                    meeting_id=meeting_id,
                    text=str(question["question"]),
                    reference_answer=str(question["groundtruth-answer"]),
                    question_type=question_type,
                    answer_position=answer_position,
                )
            )

    return DatasetSplit(tuple(meetings), tuple(queries))
