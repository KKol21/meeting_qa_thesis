from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Any, Sequence

from .artifacts import write_json
from .chunking.lumber import LUMBERCHUNKER_INSTRUCTIONS
from .config import Config
from .evaluation.elitr import ELITR_JUDGE_RUBRIC, ELITR_JUDGE_TASK
from .generation import ANSWER_INSTRUCTIONS


_PACKAGES = (
    "meeting-qa-chunking",
    "sentence-transformers",
    "transformers",
    "torch",
    "rouge-score",
)


def _digest_files(paths: Sequence[Path]) -> str:
    digest = sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hash_path(path: str | Path) -> str:
    source = Path(path).resolve()
    files = tuple(item for item in source.rglob("*") if item.is_file()) if source.is_dir() else (source,)
    return _digest_files(files)


def build_manifest(
    config: Config,
    config_path: str | Path,
    *,
    data_paths: Sequence[str | Path],
    selection: dict[str, Any],
) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    packages: dict[str, str] = {}
    for package in _PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"

    core = {
        "config": asdict(config),
        "config_hash": hash_path(config_path),
        "code_hash": _digest_files(tuple(package_root.rglob("*.py"))),
        "data": {str(Path(path).resolve()): hash_path(path) for path in data_paths},
        "packages": packages,
        "prompts": {
            "lumberchunker": sha256(
                LUMBERCHUNKER_INSTRUCTIONS.encode("utf-8")
            ).hexdigest(),
            "answer": sha256(ANSWER_INSTRUCTIONS.encode("utf-8")).hexdigest(),
            "elitr_judge": sha256(
                f"{ELITR_JUDGE_TASK}\n{ELITR_JUDGE_RUBRIC}".encode("utf-8")
            ).hexdigest(),
        },
        "selection": selection,
    }
    fingerprint = sha256(
        json.dumps(core, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **core,
    }


def ensure_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(run_dir) / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError(
                f"{path} belongs to a different configuration; choose a new run directory"
            )
        return
    write_json(path, manifest)
