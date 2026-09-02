from __future__ import annotations

from ..schema import Chunk
from ..tokenization import TokenizedTranscript


def turn_packed_chunks(
    transcript: TokenizedTranscript,
    chunk_size: int,
) -> tuple[Chunk, ...]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    ranges: list[tuple[int, int]] = []
    start = 0
    size = 0
    for index, item in enumerate(transcript.turns):
        turn_size = item.source_span.length
        if index > start and size + turn_size > chunk_size:
            ranges.append((start, index))
            start = index
            size = 0
        size += turn_size
    ranges.append((start, len(transcript.turns)))

    return tuple(
        transcript.chunk_from_turns(start, end, "turn_packed", index)
        for index, (start, end) in enumerate(ranges)
    )

