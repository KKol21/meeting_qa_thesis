"""Sequential LumberChunker control flow, independent of any model backend."""

from collections.abc import Callable
from pathlib import Path

from .artifacts import read_segmentation
from .chunking import Chunk
from .lumber_prompt import (
    build_prompt,
    build_retry_prompt,
    build_window,
    estimate_tokens,
    parse_boundary,
    render_document,
)
from .qmsum import Meeting, Turn


BoundaryChooser = Callable[[str], str]
DecisionRecorder = Callable[[str], None]


def load_lumber_chunks(path: Path, meeting: Meeting) -> list[Chunk]:
    """Recreate chunks from a saved Lumber segmentation."""

    result = read_segmentation(path)
    if result.meeting_id != meeting.id:
        raise ValueError("Lumber result belongs to a different meeting")

    chunks = [
        Chunk(
            index=item.index,
            turns=meeting.turns[item.start_turn : item.end_turn + 1],
        )
        for item in result.chunks
    ]
    covered_turns = [turn.id for chunk in chunks for turn in chunk.turns]
    if covered_turns != list(range(len(meeting.turns))):
        raise ValueError("Lumber chunks do not cover every turn exactly once")
    return chunks


def lumber_chunks(
    turns: list[Turn],
    choose_boundary: BoundaryChooser,
    target_tokens: int = 550,
    max_boundaries: int | None = None,
    record_decision: DecisionRecorder | None = None,
) -> list[Chunk]:
    """Segment turns, optionally stopping after a number of model decisions."""

    if any(turn.id != index for index, turn in enumerate(turns)):
        raise ValueError("Turn IDs must be contiguous and zero-based")
    if max_boundaries is not None and max_boundaries < 1:
        raise ValueError("max_boundaries must be positive")

    chunks: list[Chunk] = []
    start = 0

    while start < len(turns):
        window = build_window(turns, start, target_tokens)
        window_tokens = estimate_tokens(render_document(window))

        if len(window) < 2 or window_tokens <= target_tokens:
            chunks.append(Chunk(index=len(chunks), turns=turns[start:]))
            break

        response = choose_boundary(build_prompt(window))
        try:
            boundary = parse_boundary(response, window)
        except ValueError:
            response = choose_boundary(build_retry_prompt(window, response))
            boundary = parse_boundary(response, window)
        if record_decision is not None:
            record_decision(response)
        chunks.append(Chunk(index=len(chunks), turns=turns[start:boundary]))
        start = boundary

        if max_boundaries is not None and len(chunks) >= max_boundaries:
            break

    return chunks
