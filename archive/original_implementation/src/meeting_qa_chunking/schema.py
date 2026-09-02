from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True, order=True)
class Span:
    """Half-open source-token span: ``start <= token < end``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid span [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Turn:
    id: int
    speaker: str
    text: str

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("Turn id must be non-negative")

    def render(self) -> str:
        speaker = self.speaker.strip() or "Unknown speaker"
        return f"[T{self.id:04d}] {speaker}: {self.text.strip()}"


@dataclass(frozen=True, slots=True)
class Meeting:
    dataset: str
    split: str
    id: str
    turns: tuple[Turn, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError(f"Meeting {self.id} has no turns")
        actual = tuple(turn.id for turn in self.turns)
        expected = tuple(range(len(self.turns)))
        if actual != expected:
            raise ValueError(f"Meeting {self.id} must use contiguous zero-based turn ids")


@dataclass(frozen=True, slots=True)
class Query:
    dataset: str
    split: str
    id: str
    meeting_id: str
    text: str
    reference_answer: str
    gold_turn_ranges: tuple[tuple[int, int], ...] = ()
    question_type: str | None = None
    answer_position: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(f"Query {self.id} has no text")
        for start, end in self.gold_turn_ranges:
            if start < 0 or end < start:
                raise ValueError(f"Invalid inclusive turn range [{start}, {end}]")


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    meeting_id: str
    method: str
    text: str
    source_spans: tuple[Span, ...]
    turn_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.source_spans:
            raise ValueError(f"Chunk {self.id} has no source spans")
        if not self.text.strip():
            raise ValueError(f"Chunk {self.id} has no text")

    @property
    def source_token_count(self) -> int:
        return sum(span.length for span in self.source_spans)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    meetings: tuple[Meeting, ...]
    queries: tuple[Query, ...]

    def __post_init__(self) -> None:
        meeting_ids = {meeting.id for meeting in self.meetings}
        if len(meeting_ids) != len(self.meetings):
            raise ValueError("Meeting ids must be unique within a split")
        missing = {query.meeting_id for query in self.queries} - meeting_ids
        if missing:
            raise ValueError(f"Queries reference missing meetings: {sorted(missing)}")
