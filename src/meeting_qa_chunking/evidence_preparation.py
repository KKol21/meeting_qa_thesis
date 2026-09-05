"""Prepare oracle and retrieved evidence for answering and reporting."""

from collections.abc import Iterable
from pathlib import Path

from .artifacts import read_retrieval
from .chunking import (
    Chunk,
    chunk_turn_packed,
    chunk_word_packed,
    turn_word_count,
)
from .evidence import reconstruct_evidence, render_evidence, render_gold_evidence
from .lumber import load_lumber_chunks
from .qmsum import Meeting


def build_chunk_sets(
    meeting: Meeting,
    lumber_path: Path | None,
    turn_packed_max_words: int,
    word_packed_max_words: int,
    chunkers: Iterable[str] = ("turn_packed", "word_packed", "lumber"),
) -> dict[str, list[Chunk]]:
    """Build the requested chunk views."""

    builders = {
        "turn_packed": lambda: chunk_turn_packed(
            meeting.turns, turn_packed_max_words
        ),
        "word_packed": lambda: chunk_word_packed(
            meeting.turns, word_packed_max_words
        ),
    }
    built = {}
    for name in chunkers:
        if name == "lumber":
            if lumber_path is None:
                raise ValueError("Lumber chunks require a segmentation path")
            built[name] = load_lumber_chunks(lumber_path, meeting)
        elif name in builders:
            built[name] = builders[name]()
        else:
            raise ValueError(f"Unknown chunker: {name}")
    return built


def prepare_oracle_evidence(meeting: Meeting) -> list[dict[str, object]]:
    prepared = []
    for question in meeting.questions:
        text, turn_ids = render_gold_evidence(question, meeting)
        prepared.append(
            {
                "text": text,
                "metadata": {
                    "gold_turn_ranges": question.relevant_turn_ranges,
                    "evidence_words": sum(
                        turn_word_count(meeting.turns[turn_id])
                        for turn_id in turn_ids
                    ),
                },
            }
        )
    return prepared


def prepare_retrieved_evidence(
    meeting: Meeting,
    retrieval_path: Path,
    lumber_dir: Path,
    requested_conditions: list[str] | None,
) -> tuple[dict[str, dict[str, object]], list[dict[str, dict[str, object]]]]:
    retrieval = read_retrieval(retrieval_path)
    if retrieval["meeting_id"] != meeting.id:
        raise ValueError("Retrieval result belongs to a different meeting")
    conditions = retrieval["configurations"]
    selected_names = requested_conditions or list(conditions)
    unknown = [name for name in selected_names if name not in conditions]
    if unknown:
        raise ValueError(f"Unknown retrieval conditions: {', '.join(unknown)}")
    selected_conditions = {name: conditions[name] for name in selected_names}

    saved_questions = retrieval["questions"]
    if not isinstance(saved_questions, list) or len(saved_questions) != len(
        meeting.questions
    ):
        raise ValueError("Retrieval result has the wrong number of questions")

    chunkers = tuple(dict.fromkeys(
        condition["chunker"] for condition in selected_conditions.values()
    ))
    chunk_sets = build_chunk_sets(
        meeting,
        lumber_dir / f"{meeting.id}.json" if "lumber" in chunkers else None,
        retrieval["chunking"]["turn_packed_max_words"],
        retrieval["chunking"]["word_packed_max_words"],
        chunkers,
    )
    evidence_order = retrieval["evidence_order"]
    prepared = []
    for question_index, question_result in enumerate(saved_questions):
        if question_result["question_index"] != question_index:
            raise ValueError("Retrieval question indices are not contiguous")
        if question_result["question"] != meeting.questions[question_index].text:
            raise ValueError("Retrieval result contains a different question")
        question_evidence = {}
        for name, condition in selected_conditions.items():
            saved = question_result["results"][name]
            evidence = reconstruct_evidence(
                saved,
                chunk_sets[condition["chunker"]],
                condition["evidence_words"],
            )
            question_evidence[name] = {
                "text": render_evidence(evidence, meeting, evidence_order),
                "metadata": {
                    "evidence_words": evidence.word_count,
                    "evidence_order": evidence_order,
                    "selected_chunk_indices": evidence.chunk_indices,
                    "retrieval_precision": saved["precision"],
                    "retrieval_recall": saved["recall"],
                },
            }
        prepared.append(question_evidence)
    return selected_conditions, prepared
