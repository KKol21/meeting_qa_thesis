"""A simple, non-semantic meeting chunker."""

from dataclasses import dataclass

from .qmsum import Turn


def render_turn(turn: Turn) -> str:
    return f"[{turn.id}] {turn.speaker}: {turn.text}"


def turn_word_count(turn: Turn) -> int:
    return len(turn.text.split())


@dataclass
class Chunk:
    index: int
    turns: list[Turn]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Chunk index must be non-negative")
        if not self.turns:
            raise ValueError("Chunk must contain at least one turn")

    @property
    def start_turn(self) -> int:
        return self.turns[0].id

    @property
    def end_turn(self) -> int:
        return self.turns[-1].id

    @property
    def text(self) -> str:
        return "\n".join(render_turn(turn) for turn in self.turns)

    @property
    def word_count(self) -> int:
        return sum(turn_word_count(turn) for turn in self.turns)

    def overlaps(self, start_turn: int, end_turn: int) -> bool:
        """Return whether this chunk overlaps an inclusive turn range."""

        return self.start_turn <= end_turn and start_turn <= self.end_turn


def chunk_by_word_budget(turns: list[Turn], max_words: int) -> list[Chunk]:
    """Greedily pack complete turns without using their meaning."""

    if max_words <= 0:
        raise ValueError("max_words must be positive")

    chunks: list[Chunk] = []
    current_turns: list[Turn] = []
    current_words = 0

    for turn in turns:
        turn_words = turn_word_count(turn)
        if current_turns and current_words + turn_words > max_words:
            chunks.append(Chunk(index=len(chunks), turns=current_turns))
            current_turns = []
            current_words = 0

        current_turns.append(turn)
        current_words += turn_words

    if current_turns:
        chunks.append(Chunk(index=len(chunks), turns=current_turns))

    return chunks
