from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..schema import DatasetSplit, Meeting, Query, Turn


def _records(path: Path) -> Iterable[tuple[str | None, dict[str, Any]]]:
    if path.is_dir():
        for json_path in sorted(path.glob("*.json")):
            with json_path.open(encoding="utf-8") as source:
                data = json.load(source)
            if not isinstance(data, dict):
                raise ValueError(f"Expected a JSON object in {json_path}")
            yield json_path.stem, data
        return

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield None, json.loads(line)
        return

    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    if isinstance(data, dict):
        yield path.stem, data
    elif isinstance(data, list):
        yield from ((None, record) for record in data)
    else:
        raise ValueError(f"Expected a JSON object or list in {path}")


def load_qmsum(path: str | Path, split: str) -> DatasetSplit:
    """Load QMSum specific queries; general queries have no evidence labels."""

    source_path = Path(path)
    meetings: list[Meeting] = []
    queries: list[Query] = []

    for meeting_index, (inferred_id, raw) in enumerate(_records(source_path)):
        meeting_id = str(
            raw.get("meeting_id")
            or raw.get("id")
            or inferred_id
            or f"{source_path.stem}:{meeting_index:04d}"
        )
        turns = tuple(
            Turn(
                id=turn_id,
                speaker=str(turn.get("speaker", "")),
                text=str(turn["content"]),
            )
            for turn_id, turn in enumerate(raw["meeting_transcripts"])
        )
        meeting = Meeting(
            dataset="qmsum",
            split=split,
            id=meeting_id,
            turns=turns,
        )
        meetings.append(meeting)

        for query_index, item in enumerate(raw.get("specific_query_list", [])):
            ranges = tuple(
                (int(start), int(end))
                for start, end in item["relevant_text_span"]
            )
            for start, end in ranges:
                if end >= len(turns):
                    raise ValueError(
                        f"Query {meeting_id}:{query_index} references turn {end}, "
                        f"but the meeting has {len(turns)} turns"
                    )
            queries.append(
                Query(
                    dataset="qmsum",
                    split=split,
                    id=f"{meeting_id}:q{query_index:03d}",
                    meeting_id=meeting_id,
                    text=str(item["query"]),
                    reference_answer=str(item["answer"]),
                    gold_turn_ranges=ranges,
                )
            )

    return DatasetSplit(tuple(meetings), tuple(queries))
