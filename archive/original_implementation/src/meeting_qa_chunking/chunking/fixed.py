from __future__ import annotations

from ..schema import Chunk, Span
from ..tokenization import TokenizedTranscript, Tokenizer


def fixed_token_chunks(
    transcript: TokenizedTranscript,
    tokenizer: Tokenizer,
    chunk_size: int,
) -> tuple[Chunk, ...]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, len(transcript.token_ids), chunk_size)):
        end = min(start + chunk_size, len(transcript.token_ids))
        turn_ids = tuple(
            item.turn.id
            for item in transcript.turns
            if item.source_span.start < end and start < item.source_span.end
        )
        chunks.append(
            Chunk(
                id=f"{transcript.meeting.id}:fixed:{index:04d}",
                meeting_id=transcript.meeting.id,
                method="fixed",
                text=tokenizer.decode(transcript.token_ids[start:end]),
                source_spans=(Span(start, end),),
                turn_ids=turn_ids,
            )
        )
    return tuple(chunks)

