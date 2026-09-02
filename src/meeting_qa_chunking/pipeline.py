"""Pure preparation shared by retrieval, answering, and reporting stages."""

from pathlib import Path

from .artifacts import read_retrieval
from .chunking import Chunk, chunk_by_word_budget, turn_word_count
from .evidence import reconstruct_evidence, render_evidence, render_gold_evidence
from .lumber import load_lumber_chunks
from .qmsum import Meeting


def build_chunk_sets(
    meeting: Meeting,
    lumber_path: Path,
    fixed_chunk_words: int,
) -> dict[str, list[Chunk]]:
    """Build the two chunk sets used by the existing ablation grid."""

    return {
        "fixed": chunk_by_word_budget(meeting.turns, fixed_chunk_words),
        "lumber": load_lumber_chunks(lumber_path, meeting),
    }


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

    chunk_sets = build_chunk_sets(
        meeting,
        lumber_dir / f"{meeting.id}.json",
        retrieval["fixed_chunk_words"],
    )
    prepared = []
    for question_index, question_result in enumerate(retrieval["questions"]):
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
                "text": render_evidence(evidence, meeting),
                "metadata": {
                    "evidence_words": evidence.word_count,
                    "selected_chunk_indices": evidence.chunk_indices,
                    "retrieval_precision": saved["precision"],
                    "retrieval_recall": saved["recall"],
                },
            }
        prepared.append(question_evidence)
    return selected_conditions, prepared
