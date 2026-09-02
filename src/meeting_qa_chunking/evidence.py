"""Select retrieved evidence under a fixed word budget."""

from dataclasses import dataclass

from .chunking import Chunk, render_turn, turn_word_count
from .qmsum import Meeting, Question


@dataclass
class EvidencePart:
    chunk_index: int
    turn_id: int
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Evidence:
    parts: list[EvidencePart]

    @property
    def word_count(self) -> int:
        return sum(part.word_count for part in self.parts)

    @property
    def chunk_indices(self) -> list[int]:
        return list(dict.fromkeys(part.chunk_index for part in self.parts))


def render_evidence(evidence: Evidence, meeting: Meeting) -> str:
    """Render retrieved parts with their original turn IDs and speakers."""

    return "\n".join(
        f"[{part.turn_id}] {meeting.turns[part.turn_id].speaker}: {part.text}"
        for part in evidence.parts
    )


def reconstruct_evidence(
    saved_result: dict[str, object],
    chunks: list[Chunk],
    max_words: int,
) -> Evidence:
    indices = saved_result["selected_chunk_indices"]
    if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
        raise ValueError("Invalid selected_chunk_indices in retrieval result")
    if any(index < 0 or index >= len(chunks) for index in indices):
        raise ValueError("Retrieved chunk index is out of range")

    evidence = select_evidence([(index, 0.0) for index in indices], chunks, max_words)
    if evidence.word_count != saved_result["retrieved_words"]:
        raise ValueError("Reconstructed evidence differs from retrieval result")
    return evidence


def gold_turn_ids(question: Question, meeting: Meeting) -> list[int]:
    if any(
        start < 0 or end < start or end >= len(meeting.turns)
        for start, end in question.relevant_turn_ranges
    ):
        raise ValueError("Gold evidence contains an invalid turn range")
    turn_ids = sorted(
        {
            turn_id
            for start, end in question.relevant_turn_ranges
            for turn_id in range(start, end + 1)
        }
    )
    if not turn_ids:
        raise ValueError("Question has no gold evidence")
    return turn_ids


def render_gold_evidence(question: Question, meeting: Meeting) -> tuple[str, list[int]]:
    turn_ids = gold_turn_ids(question, meeting)
    text = "\n".join(render_turn(meeting.turns[turn_id]) for turn_id in turn_ids)
    return text, turn_ids


@dataclass
class RetrievalMetrics:
    precision: float
    recall: float
    retrieved_words: int
    relevant_retrieved_words: int
    gold_words: int


def first_relevant_chunk_rank(
    ranking: list[tuple[int, float]],
    chunks: list[Chunk],
    question: Question,
) -> int | None:
    """Return the one-based rank of the first chunk overlapping gold evidence."""

    for rank, (chunk_index, _score) in enumerate(ranking, start=1):
        chunk = chunks[chunk_index]
        if any(
            chunk.overlaps(start, end)
            for start, end in question.relevant_turn_ranges
        ):
            return rank
    return None


def select_evidence(
    ranking: list[tuple[int, float]],
    chunks: list[Chunk],
    max_words: int,
) -> Evidence:
    """Take ranked chunks until the budget is full, clipping the final turn."""

    if max_words <= 0:
        raise ValueError("max_words must be positive")

    parts: list[EvidencePart] = []
    seen_turns: set[int] = set()
    remaining = max_words

    for chunk_index, _score in ranking:
        for turn in chunks[chunk_index].turns:
            if turn.id in seen_turns:
                continue
            seen_turns.add(turn.id)

            words = turn.text.split()
            selected_words = words[:remaining]
            if selected_words:
                parts.append(
                    EvidencePart(
                        chunk_index=chunk_index,
                        turn_id=turn.id,
                        text=" ".join(selected_words),
                    )
                )
                remaining -= len(selected_words)

            if remaining == 0:
                return Evidence(parts)

    return Evidence(parts)


def score_evidence(
    evidence: Evidence,
    meeting: Meeting,
    question: Question,
) -> RetrievalMetrics:
    gold_turns = {
        turn_id
        for start, end in question.relevant_turn_ranges
        for turn_id in range(start, end + 1)
    }
    gold_words = sum(
        turn_word_count(meeting.turns[turn_id]) for turn_id in gold_turns
    )
    relevant_retrieved_words = sum(
        part.word_count for part in evidence.parts if part.turn_id in gold_turns
    )
    retrieved_words = evidence.word_count

    return RetrievalMetrics(
        precision=(
            relevant_retrieved_words / retrieved_words if retrieved_words else 0.0
        ),
        recall=(relevant_retrieved_words / gold_words if gold_words else 0.0),
        retrieved_words=retrieved_words,
        relevant_retrieved_words=relevant_retrieved_words,
        gold_words=gold_words,
    )
