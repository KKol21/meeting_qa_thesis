from __future__ import annotations

from dataclasses import dataclass

from ..retrieval import Evidence
from ..schema import Query, Span
from ..tokenization import TokenizedTranscript


@dataclass(frozen=True, slots=True)
class TokenScores:
    precision: float
    recall: float
    f1: float
    overlap_tokens: int
    retrieved_tokens: int
    gold_tokens: int

    @property
    def zero_hit(self) -> bool:
        return self.overlap_tokens == 0


def _positions(spans: tuple[Span, ...]) -> set[int]:
    return {position for span in spans for position in range(span.start, span.end)}


def gold_source_spans(
    query: Query,
    transcript: TokenizedTranscript,
) -> tuple[Span, ...]:
    if query.meeting_id != transcript.meeting.id:
        raise ValueError("Query and transcript refer to different meetings")
    if not query.gold_turn_ranges:
        raise ValueError(f"Query {query.id} has no gold turn ranges")

    positions: set[int] = set()
    for start, end in query.gold_turn_ranges:
        if end >= len(transcript.turns):
            raise ValueError(f"Gold turn range [{start}, {end}] is out of bounds")
        first = transcript.turns[start].source_span.start
        last = transcript.turns[end].source_span.end
        positions.update(range(first, last))

    ordered = sorted(positions)
    spans: list[Span] = []
    start = previous = ordered[0]
    for position in ordered[1:]:
        if position != previous + 1:
            spans.append(Span(start, previous + 1))
            start = position
        previous = position
    spans.append(Span(start, previous + 1))
    return tuple(spans)


def score_evidence(
    query: Query,
    evidence: Evidence,
    transcript: TokenizedTranscript,
) -> TokenScores:
    retrieved = _positions(evidence.source_spans)
    gold = _positions(gold_source_spans(query, transcript))
    overlap = len(retrieved & gold)
    precision = overlap / len(retrieved) if retrieved else 0.0
    recall = overlap / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
    return TokenScores(precision, recall, f1, overlap, len(retrieved), len(gold))
