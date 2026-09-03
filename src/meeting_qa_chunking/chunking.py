"""Turn-packed and strictly word-packed meeting baselines."""

from dataclasses import dataclass

from .qmsum import Turn


def render_turn(turn: Turn) -> str:
    return f"[{turn.id}] {turn.speaker}: {turn.text}"


def turn_word_count(turn: Turn) -> int:
    return len(turn.text.split())


@dataclass(frozen=True)
class ChunkPart:
    """A complete turn or a word-offset fragment of one turn."""

    turn_id: int
    speaker: str
    text: str
    start_word: int = 0

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def end_word(self) -> int:
        return self.start_word + self.word_count

    @classmethod
    def from_turn(cls, turn: Turn) -> "ChunkPart":
        return cls(turn.id, turn.speaker, turn.text)


@dataclass
class Chunk:
    index: int
    parts: list[ChunkPart]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Chunk index must be non-negative")
        if not self.parts:
            raise ValueError("Chunk must contain at least one part")

    @classmethod
    def from_turns(cls, index: int, turns: list[Turn]) -> "Chunk":
        return cls(index, [ChunkPart.from_turn(turn) for turn in turns])

    @property
    def start_turn(self) -> int:
        return self.parts[0].turn_id

    @property
    def end_turn(self) -> int:
        return self.parts[-1].turn_id

    @property
    def text(self) -> str:
        # A split-turn continuation repeats its speaker attribution.
        return "\n".join(
            f"[{part.turn_id}] {part.speaker}: {part.text}"
            for part in self.parts
        )

    @property
    def word_count(self) -> int:
        return sum(part.word_count for part in self.parts)

    def overlaps(self, start_turn: int, end_turn: int) -> bool:
        """Return whether this chunk overlaps an inclusive turn range."""

        return any(start_turn <= part.turn_id <= end_turn for part in self.parts)


def chunk_turn_packed(turns: list[Turn], max_words: int) -> list[Chunk]:
    """Greedily pack complete turns under a soft word limit."""

    if max_words <= 0:
        raise ValueError("max_words must be positive")

    chunks: list[Chunk] = []
    current_turns: list[Turn] = []
    current_words = 0

    for turn in turns:
        words = turn_word_count(turn)
        if current_turns and current_words + words > max_words:
            chunks.append(Chunk.from_turns(len(chunks), current_turns))
            current_turns = []
            current_words = 0
        current_turns.append(turn)
        current_words += words

    if current_turns:
        chunks.append(Chunk.from_turns(len(chunks), current_turns))
    return chunks


def chunk_word_packed(turns: list[Turn], max_words: int) -> list[Chunk]:
    """Create hard-size chunks, splitting turns and repeating speaker labels."""

    if max_words <= 0:
        raise ValueError("max_words must be positive")

    chunks: list[Chunk] = []
    parts: list[ChunkPart] = []
    used = 0

    def flush() -> None:
        nonlocal parts, used
        if parts:
            chunks.append(Chunk(len(chunks), parts))
            parts = []
            used = 0

    for turn in turns:
        words = turn.text.split()
        if not words:
            parts.append(ChunkPart.from_turn(turn))
            continue

        offset = 0
        while offset < len(words):
            if used == max_words:
                flush()
            take = min(max_words - used, len(words) - offset)
            parts.append(
                ChunkPart(
                    turn_id=turn.id,
                    speaker=turn.speaker,
                    text=" ".join(words[offset : offset + take]),
                    start_word=offset,
                )
            )
            offset += take
            used += take
            if used == max_words and offset < len(words):
                flush()

    flush()
    return chunks
