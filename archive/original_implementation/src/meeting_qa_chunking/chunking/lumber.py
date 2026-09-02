from __future__ import annotations

from dataclasses import dataclass
import re

from ..models import TextModel
from ..schema import Chunk, Turn
from ..tokenization import TokenizedTranscript


LUMBERCHUNKER_INSTRUCTIONS = (
    "You will receive as input an English document with paragraphs identified by "
    "'ID XXXX: <text>'.\n\n"
    "Task: Find the first paragraph (not the first one) where the content clearly "
    "changes compared to the previous paragraphs.\n\n"
    "Output: Return the ID of the paragraph with the content shift as in the "
    "exemplified format: 'Answer: ID XXXX'.\n"
    "Additional Considerations: Avoid very long groups of paragraphs. Aim for a "
    "good balance between identifying content shifts and keeping groups manageable."
)
_ANSWER = re.compile(r"Answer:\s*ID\s*:?[ ]*(\d+)", re.IGNORECASE)


BoundaryModel = TextModel


class LumberChunkerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BoundaryDecision:
    window_turn_ids: tuple[int, ...]
    boundary_turn_id: int
    response: str
    prompt: str


@dataclass(frozen=True, slots=True)
class LumberResult:
    chunks: tuple[Chunk, ...]
    decisions: tuple[BoundaryDecision, ...]


def estimate_tokens(text: str) -> int:
    """Token estimate used by the public LumberChunker implementation."""

    return round(1.2 * len(text.split()))


def _render_turn(turn: Turn) -> str:
    speaker = turn.speaker.strip() or "Unknown speaker"
    return f"ID {turn.id + 1:04d}: {speaker}: {turn.text.strip()}"


def _render_document(turns: tuple[Turn, ...]) -> str:
    return "\n".join(_render_turn(turn) for turn in turns)


def build_prompt(turns: tuple[Turn, ...]) -> str:
    return f"{LUMBERCHUNKER_INSTRUCTIONS}\n\nDocument:\n{_render_document(turns)}"


def parse_boundary(response: str, window: tuple[Turn, ...]) -> int:
    match = _ANSWER.search(response)
    if not match:
        raise LumberChunkerError(f"Could not parse boundary from {response!r}")
    boundary = int(match.group(1)) - 1
    valid = {turn.id for turn in window[1:]}
    if boundary not in valid:
        raise LumberChunkerError(
            f"Boundary turn {boundary} is not a non-initial turn in the window"
        )
    return boundary


class LumberChunker:
    """Sequential local-window segmentation from Duarte et al. (2024)."""

    def __init__(
        self,
        model: BoundaryModel,
        *,
        window_tokens: int = 550,
        temperature: float = 0.1,
        max_attempts: int = 2,
    ) -> None:
        if window_tokens <= 0 or max_attempts <= 0:
            raise ValueError("window_tokens and max_attempts must be positive")
        self.model = model
        self.window_tokens = window_tokens
        self.temperature = temperature
        self.max_attempts = max_attempts

    def segment(self, transcript: TokenizedTranscript) -> LumberResult:
        turns = transcript.meeting.turns
        start = 0
        ranges: list[tuple[int, int]] = []
        decisions: list[BoundaryDecision] = []

        while start < len(turns):
            window = self._window(turns, start)
            window_size = estimate_tokens(_render_document(window))
            if len(window) < 2 or window_size <= self.window_tokens:
                ranges.append((start, len(turns)))
                break

            prompt = build_prompt(window)
            last_error: LumberChunkerError | None = None
            for _ in range(self.max_attempts):
                response = self.model.complete(prompt, temperature=self.temperature)
                try:
                    boundary = parse_boundary(response, window)
                    break
                except LumberChunkerError as error:
                    last_error = error
            else:
                raise LumberChunkerError(
                    f"No valid boundary after {self.max_attempts} attempts"
                ) from last_error

            ranges.append((start, boundary))
            decisions.append(
                BoundaryDecision(
                    tuple(turn.id for turn in window), boundary, response, prompt
                )
            )
            start = boundary

        chunks = tuple(
            transcript.chunk_from_turns(start, end, "lumber", index)
            for index, (start, end) in enumerate(ranges)
        )
        return LumberResult(chunks, tuple(decisions))

    def _window(self, turns: tuple[Turn, ...], start: int) -> tuple[Turn, ...]:
        selected: list[Turn] = []
        for turn in turns[start:]:
            selected.append(turn)
            if (
                estimate_tokens(_render_document(tuple(selected))) > self.window_tokens
                and len(selected) >= 2
            ):
                break
        return tuple(selected)
