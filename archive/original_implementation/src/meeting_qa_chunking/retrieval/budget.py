from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..schema import Span
from ..tokenization import TokenizedTranscript, Tokenizer
from .dense import RankedChunk


@dataclass(frozen=True, slots=True)
class Evidence:
    text: str
    source_spans: tuple[Span, ...]
    selected_chunk_ids: tuple[str, ...]

    @property
    def source_token_count(self) -> int:
        return sum(span.length for span in self.source_spans)


def _merge_positions(positions: set[int]) -> tuple[Span, ...]:
    ordered = sorted(positions)
    if not ordered:
        return ()

    spans: list[Span] = []
    start = previous = ordered[0]
    for position in ordered[1:]:
        if position != previous + 1:
            spans.append(Span(start, previous + 1))
            start = position
        previous = position
    spans.append(Span(start, previous + 1))
    return tuple(spans)


def project_evidence(
    ranking: Sequence[RankedChunk],
    transcript: TokenizedTranscript,
    tokenizer: Tokenizer,
    budget: int,
) -> Evidence:
    """Take ranked chunks until exactly ``budget`` unique source tokens are exposed."""

    if budget <= 0:
        raise ValueError("budget must be positive")

    target = min(budget, len(transcript.token_ids))
    selected: set[int] = set()
    selected_chunk_ids: list[str] = []

    for item in ranking:
        chunk = item.chunk
        if chunk.meeting_id != transcript.meeting.id:
            raise ValueError(f"Chunk {chunk.id} belongs to another meeting")

        before = len(selected)
        for span in chunk.source_spans:
            if span.end > len(transcript.token_ids):
                raise ValueError(f"Chunk {chunk.id} has an out-of-range source span")
            for position in range(span.start, span.end):
                if len(selected) == target:
                    break
                selected.add(position)
            if len(selected) == target:
                break

        if len(selected) > before:
            selected_chunk_ids.append(chunk.id)
        if len(selected) == target:
            break

    if len(selected) != target:
        raise ValueError("Ranked chunks do not cover the requested source-token budget")

    spans = _merge_positions(selected)
    text = "\n".join(
        tokenizer.decode(transcript.token_ids[span.start : span.end]) for span in spans
    )
    return Evidence(text, spans, tuple(selected_chunk_ids))
