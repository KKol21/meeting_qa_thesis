"""Build local LumberChunker windows and prompts."""

import re

from .prompt_files import load_prompt
from .qmsum import Turn


LUMBERCHUNKER_INSTRUCTIONS = load_prompt("lumberchunker.txt")

BOUNDARY_PATTERN = re.compile(r"Answer:\s*ID\s+(\d+)", re.IGNORECASE)


def render_turn(turn: Turn) -> str:
    speaker = turn.speaker.strip() or "Unknown speaker"
    return f"ID {turn.id:04d}: {speaker}: {turn.text.strip()}"


def render_document(turns: list[Turn]) -> str:
    return "\n".join(render_turn(turn) for turn in turns)


def estimate_tokens(text: str) -> int:
    """Approximation used by the original LumberChunker implementation."""

    return round(1.2 * len(text.split()))


def build_window(
    turns: list[Turn],
    start_turn: int,
    target_tokens: int = 550,
) -> list[Turn]:
    """Take consecutive turns until the local window exceeds the target."""

    if not 0 <= start_turn < len(turns):
        raise ValueError("start_turn is outside the transcript")
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    window: list[Turn] = []
    for turn in turns[start_turn:]:
        window.append(turn)
        if len(window) >= 2 and estimate_tokens(render_document(window)) > target_tokens:
            break

    return window


def build_prompt(window: list[Turn]) -> str:
    return f"{LUMBERCHUNKER_INSTRUCTIONS}\n\nDocument:\n{render_document(window)}"


def build_retry_prompt(window: list[Turn], invalid_response: str) -> str:
    """Repeat the original task while constraining one invalid answer."""

    valid_ids = ", ".join(f"ID {turn.id:04d}" for turn in window[1:])
    return (
        f"{build_prompt(window)}\n\n"
        f"Your previous response was invalid: {invalid_response!r}\n"
        f"The first paragraph, ID {window[0].id:04d}, cannot be selected. "
        f"Choose exactly one of: {valid_ids}.\n"
        "Return only 'Answer: ID XXXX'."
    )


def parse_boundary(response: str, window: list[Turn]) -> int:
    """Extract a valid next-chunk start from a model response."""

    match = BOUNDARY_PATTERN.search(response)
    if not match:
        raise ValueError(f"Could not parse LumberChunker response: {response!r}")

    boundary = int(match.group(1))
    valid_boundaries = {turn.id for turn in window[1:]}
    if boundary not in valid_boundaries:
        raise ValueError(
            f"Boundary {boundary} must be a non-initial turn in the local window"
        )

    return boundary
