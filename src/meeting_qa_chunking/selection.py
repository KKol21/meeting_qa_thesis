"""Deterministic meeting selection for experiment presets."""

import random
from pathlib import Path


def select_meeting_paths(
    data_dir: Path,
    count: int,
    seed: int,
    meeting_ids: list[str] | None = None,
) -> list[Path]:
    paths = {path.stem: path for path in data_dir.glob("*.json")}
    if meeting_ids:
        missing = [meeting_id for meeting_id in meeting_ids if meeting_id not in paths]
        if missing:
            raise ValueError(f"Meetings not found: {', '.join(missing)}")
        return [paths[meeting_id] for meeting_id in meeting_ids]

    if count <= 0 or count > len(paths):
        raise ValueError(f"count must be between 1 and {len(paths)}")
    selected = sorted(paths.values(), key=lambda path: path.name)
    random.Random(seed).shuffle(selected)
    return selected[:count]
