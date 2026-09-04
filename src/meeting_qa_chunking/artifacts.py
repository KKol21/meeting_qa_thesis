"""Read, write, and fingerprint experiment artifacts."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_VERSION = 2


@dataclass(frozen=True)
class SavedChunk:
    index: int
    start_turn: int
    end_turn: int


@dataclass(frozen=True)
class SavedSegmentation:
    """The fields needed to reconstruct a saved Lumber segmentation."""

    meeting_id: str
    chunks: tuple[SavedChunk, ...]
    experiment_version: int | None


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


@dataclass(frozen=True)
class ArtifactSchema:
    """Minimal shape required to consume one historical result type."""

    name: str
    required_fields: tuple[str, ...]


RETRIEVAL_SCHEMA = ArtifactSchema(
    "retrieval",
    (
        "meeting_id",
        "configurations",
        "chunking",
        "evidence_order",
        "questions",
    ),
)
ANSWER_SCHEMA = ArtifactSchema(
    "answers",
    ("meeting_id", "source", "conditions", "answer_model", "questions"),
)
ANSWER_SUMMARY_SCHEMA = ArtifactSchema(
    "answer summary",
    ("source", "answer_model", "meeting_ids", "conditions"),
)


def sha256_json(value: object) -> str:
    content = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_provenance(
    stage: str,
    config: dict[str, object],
    inputs: dict[str, Path],
    preset_path: Path,
) -> dict[str, object]:
    """Describe exactly which effective settings and files produced a stage."""

    input_records = {
        name: {"file": path.name, "sha256": sha256_file(path)}
        for name, path in sorted(inputs.items())
    }
    config_hash = sha256_json(config)
    input_hash = sha256_json(input_records)
    fingerprint = sha256_json(
        {"stage": stage, "config_hash": config_hash, "input_hash": input_hash}
    )
    return {
        "stage": stage,
        "config": config,
        "config_hash": config_hash,
        "inputs": input_records,
        "input_hash": input_hash,
        "fingerprint": fingerprint,
        # Useful for auditing, but unrelated preset edits do not invalidate work.
        "preset": {"file": preset_path.name, "sha256": sha256_file(preset_path)},
    }


def write_json(path: Path, value: object) -> None:
    """Atomically write an experiment artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(path)


def questions_complete(
    artifact: dict[str, Any],
    meeting_id: str,
    question_texts: list[str],
    conditions: set[str],
) -> bool:
    """Check the repeated per-question shape used by retrieval and answers."""

    questions = artifact.get("questions")
    return (
        artifact.get("meeting_id") == meeting_id
        and isinstance(questions, list)
        and len(questions) == len(question_texts)
        and all(
            item.get("question_index") == index
            and item.get("question") == question_texts[index]
            and set(item.get("results", {})) == conditions
            for index, item in enumerate(questions)
        )
    )


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object with a useful error for other top-level values."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_artifact(path: Path, schema: ArtifactSchema) -> dict[str, Any]:
    """Read either a current or legacy result matching a minimal schema."""

    value = read_json_object(path)
    missing = [field for field in schema.required_fields if field not in value]
    if missing:
        raise ValueError(
            f"Invalid {schema.name} artifact; missing: {', '.join(missing)}"
        )
    return value


def read_retrieval(path: Path) -> dict[str, Any]:
    return read_artifact(path, RETRIEVAL_SCHEMA)


def read_answers(path: Path) -> dict[str, Any]:
    return read_artifact(path, ANSWER_SCHEMA)


def read_answer_summary(path: Path) -> dict[str, Any]:
    return read_artifact(path, ANSWER_SUMMARY_SCHEMA)


def read_segmentation(path: Path) -> SavedSegmentation:
    """Read current or legacy Lumber JSON.

    Legacy files may omit ``experiment_version`` and per-chunk ``index``.
    The adapter supplies contiguous indices in memory and never modifies the file.
    """

    raw = read_json_object(path)

    meeting_id = raw.get("meeting_id")
    if not isinstance(meeting_id, str) or not meeting_id:
        raise ValueError("Segmentation artifact has an invalid meeting_id")

    version = raw.get("experiment_version")
    if version is not None:
        version = _integer(version, "experiment_version")

    saved_chunks = raw.get("chunks")
    if not isinstance(saved_chunks, list):
        raise ValueError("Segmentation artifact has invalid chunks")

    chunks = []
    for expected_index, item in enumerate(saved_chunks):
        if not isinstance(item, dict):
            raise ValueError("Each saved chunk must be a JSON object")
        index = _integer(item.get("index", expected_index), "chunk index")
        if index != expected_index:
            raise ValueError("Lumber chunk indices must be contiguous and zero-based")
        start_turn = _integer(item.get("start_turn"), "start_turn")
        end_turn = _integer(item.get("end_turn"), "end_turn")
        if start_turn < 0 or end_turn < start_turn:
            raise ValueError("Saved chunk contains an invalid turn range")
        chunks.append(SavedChunk(index, start_turn, end_turn))

    return SavedSegmentation(meeting_id, tuple(chunks), version)
