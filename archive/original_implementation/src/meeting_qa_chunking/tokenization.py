from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .schema import Chunk, Meeting, Span, Turn


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class WhitespaceTokenizer:
    """Small deterministic tokenizer for tests and dependency-free development."""

    def __init__(self) -> None:
        self._tokens: list[str] = []

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in text.split():
            ids.append(len(self._tokens))
            self._tokens.append(token)
        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(self._tokens[token_id] for token_id in token_ids)


class HuggingFaceTokenizer:
    def __init__(self, model_name: str, *, revision: str | None = None) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Install the experiment dependencies: pip install -e '.[experiment]'"
            ) from error
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, revision=revision, local_files_only=True
            )
        except OSError:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, revision=revision
            )

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


@dataclass(frozen=True, slots=True)
class TokenizedTurn:
    turn: Turn
    source_span: Span
    token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TokenizedTranscript:
    meeting: Meeting
    turns: tuple[TokenizedTurn, ...]
    token_ids: tuple[int, ...]

    def chunk_from_turns(self, start: int, end: int, method: str, index: int) -> Chunk:
        selected = self.turns[start:end]
        if not selected:
            raise ValueError("Cannot create an empty chunk")
        return Chunk(
            id=f"{self.meeting.id}:{method}:{index:04d}",
            meeting_id=self.meeting.id,
            method=method,
            text="\n".join(item.turn.render() for item in selected),
            source_spans=(Span(selected[0].source_span.start, selected[-1].source_span.end),),
            turn_ids=tuple(item.turn.id for item in selected),
        )


def tokenize_meeting(meeting: Meeting, tokenizer: Tokenizer) -> TokenizedTranscript:
    cursor = 0
    all_tokens: list[int] = []
    tokenized_turns: list[TokenizedTurn] = []
    for turn in meeting.turns:
        token_ids = tuple(tokenizer.encode(turn.render()))
        if not token_ids:
            raise ValueError(f"Tokenizer produced no tokens for turn {turn.id}")
        span = Span(cursor, cursor + len(token_ids))
        tokenized_turns.append(TokenizedTurn(turn, span, token_ids))
        all_tokens.extend(token_ids)
        cursor = span.end
    return TokenizedTranscript(meeting, tuple(tokenized_turns), tuple(all_tokens))
