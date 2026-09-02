from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import write_json
from .evaluation.bootstrap import PairedScore, paired_meeting_bootstrap


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Summarize one evaluated QMSum or ELITR run and write ``summary.json``."""

    root = Path(run_dir)
    manifest = _read_json(root / "manifest.json")
    retrieval = _read_jsonl(root / "retrieval.jsonl")
    metrics = _read_jsonl(root / "metrics.jsonl")

    selection = _mapping(manifest, "selection")
    config = _mapping(manifest, "config")
    dataset = _text(selection, "dataset")
    if dataset not in {"qmsum", "elitr"}:
        raise ValueError(f"Unsupported dataset in manifest: {dataset!r}")
    _validate_dataset(dataset, retrieval, metrics)
    _validate_completeness(selection, retrieval, metrics)

    samples = int(_mapping(config, "evaluation")["bootstrap_samples"])
    seed = int(config["seed"])
    answer_budget = int(selection["answer_budget"])
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    if dataset == "qmsum":
        primary_budget = int(_mapping(config, "retrieval")["primary_budget"])
        summary = _summarize_qmsum(
            manifest,
            retrieval,
            metrics,
            primary_budget=primary_budget,
            answer_budget=answer_budget,
            samples=samples,
            seed=seed,
        )
    else:
        summary = _summarize_elitr(
            manifest,
            metrics,
            answer_budget=answer_budget,
            samples=samples,
            seed=seed,
        )

    write_json(root / "summary.json", summary)
    return summary


def _summarize_qmsum(
    manifest: Mapping[str, Any],
    retrieval: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    *,
    primary_budget: int,
    answer_budget: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    retrieval_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in retrieval:
        retrieval_groups[(_text(record, "method"), int(record["budget"]))].append(
            _mapping(record, "metrics")
        )

    retrieval_summary: dict[str, dict[str, Any]] = defaultdict(dict)
    for (method, budget), rows in sorted(retrieval_groups.items()):
        retrieval_summary[method][str(budget)] = {
            "queries": len(rows),
            "precision": _mean(_number(row, "precision") for row in rows),
            "recall": _mean(_number(row, "recall") for row in rows),
            "f1": _mean(_number(row, "f1") for row in rows),
            "zero_hit_rate": _mean(float(bool(row["zero_hit"])) for row in rows),
        }

    answer_rows = [record for record in metrics if int(record["budget"]) == answer_budget]
    if not answer_rows:
        raise ValueError(f"No QMSum metrics at answer budget {answer_budget}")
    rouge_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in answer_rows:
        rouge_groups[_text(record, "method")].append(_mapping(record, "rouge"))
    rouge_summary = {
        method: {
            "queries": len(rows),
            "rouge1": _mean(_number(row, "rouge1") for row in rows),
            "rouge2": _mean(_number(row, "rouge2") for row in rows),
            "rougeL": _mean(_number(row, "rougeL") for row in rows),
        }
        for method, rows in sorted(rouge_groups.items())
    }

    bootstrap = {
        "settings": {"samples": samples, "seed": seed},
        "retrieval_recall": {
            "budget": primary_budget,
            **_comparisons(
                retrieval,
                value=lambda row: _number(_mapping(row, "metrics"), "recall"),
                budget=primary_budget,
                samples=samples,
                seed=seed,
            ),
        },
        "rougeL": {
            "budget": answer_budget,
            **_comparisons(
                metrics,
                value=lambda row: _number(_mapping(row, "rouge"), "rougeL"),
                budget=answer_budget,
                samples=samples,
                seed=seed,
            ),
        },
    }
    selection = _mapping(manifest, "selection")
    return {
        "dataset": "qmsum",
        "split": selection.get("split"),
        "retrieval": dict(retrieval_summary),
        "rouge": {"answer_budget": answer_budget, "by_method": rouge_summary},
        "bootstrap": bootstrap,
    }


def _summarize_elitr(
    manifest: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    *,
    answer_budget: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rows = [record for record in metrics if int(record["budget"]) == answer_budget]
    if not rows:
        raise ValueError(f"No ELITR metrics at answer budget {answer_budget}")
    by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in rows:
        by_method[_text(record, "method")].append(record)

    judge = {}
    for method, method_rows in sorted(by_method.items()):
        judge[method] = {
            "queries": len(method_rows),
            "mean_score": _mean(_number(row, "score") for row in method_rows),
            "by_question_type": _diagnostic(method_rows, "question_type"),
            "by_answer_position": _diagnostic(method_rows, "answer_position"),
        }

    selection = _mapping(manifest, "selection")
    return {
        "dataset": "elitr",
        "split": selection.get("split"),
        "judge": {"answer_budget": answer_budget, "by_method": judge},
        "bootstrap": {
            "settings": {"samples": samples, "seed": seed},
            "score": {
                "budget": answer_budget,
                **_comparisons(
                    metrics,
                    value=lambda row: _number(row, "score"),
                    budget=answer_budget,
                    samples=samples,
                    seed=seed,
                ),
            },
        },
    }


def _comparisons(
    records: Sequence[Mapping[str, Any]],
    *,
    value: Any,
    budget: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    selected = [record for record in records if int(record["budget"]) == budget]
    indexed: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = defaultdict(dict)
    for record in selected:
        method = _text(record, "method")
        key = _record_identity(record)
        if key in indexed[method]:
            raise ValueError(f"Duplicate result for {method} and query {key[1]}")
        indexed[method][key] = record

    output = {}
    for baseline in ("fixed", "turn_packed"):
        if "lumber" not in indexed or baseline not in indexed:
            continue
        candidate_keys = set(indexed["lumber"])
        baseline_keys = set(indexed[baseline])
        if candidate_keys != baseline_keys:
            raise ValueError(f"Cannot pair lumber and {baseline}: query sets differ")
        paired = [
            PairedScore(
                meeting_id=meeting_id,
                candidate=float(value(indexed["lumber"][key])),
                baseline=float(value(indexed[baseline][key])),
            )
            for key in sorted(candidate_keys)
            for meeting_id, _query_id in (key,)
        ]
        output[f"lumber_vs_{baseline}"] = asdict(
            paired_meeting_bootstrap(paired, samples=samples, seed=seed)
        )
    return output


def _record_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    query = record.get("query")
    if isinstance(query, dict):
        return _text(query, "meeting_id"), _text(query, "id")
    return _text(record, "meeting_id"), _text(record, "query_id")


def _diagnostic(
    records: Sequence[Mapping[str, Any]], label: str
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[_text(record, label)].append(_number(record, "score"))
    return {
        name: {"queries": len(values), "mean_score": _mean(values)}
        for name, values in sorted(grouped.items())
    }


def _validate_dataset(
    dataset: str,
    retrieval: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
) -> None:
    retrieval_datasets = {
        _text(_mapping(record, "query"), "dataset") for record in retrieval
    }
    metric_datasets = {_text(record, "dataset") for record in metrics}
    found = retrieval_datasets | metric_datasets
    if found != {dataset}:
        raise ValueError(
            f"Run artifacts must contain only manifest dataset {dataset!r}; found {sorted(found)}"
        )


def _validate_completeness(
    selection: Mapping[str, Any],
    retrieval: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
) -> None:
    query_ids = _text_sequence(selection, "query_ids")
    methods = _text_sequence(selection, "methods")
    budgets = tuple(int(value) for value in _sequence(selection, "retrieval_budgets"))
    answer_budget = int(selection["answer_budget"])

    expected_retrieval = {
        (query_id, method, budget)
        for query_id in query_ids
        for method in methods
        for budget in budgets
    }
    actual_retrieval = [
        (_record_identity(record)[1], _text(record, "method"), int(record["budget"]))
        for record in retrieval
    ]
    _require_exact_keys("retrieval", expected_retrieval, actual_retrieval)

    expected_metrics = {
        (query_id, method, answer_budget)
        for query_id in query_ids
        for method in methods
    }
    actual_metrics = [
        (_record_identity(record)[1], _text(record, "method"), int(record["budget"]))
        for record in metrics
    ]
    _require_exact_keys("metrics", expected_metrics, actual_metrics)


def _require_exact_keys(
    artifact: str,
    expected: set[tuple[str, str, int]],
    actual: Sequence[tuple[str, str, int]],
) -> None:
    actual_set = set(actual)
    duplicates = len(actual) - len(actual_set)
    missing = expected - actual_set
    extra = actual_set - expected
    if duplicates or missing or extra:
        raise ValueError(
            f"Incomplete {artifact} artifact: {len(missing)} missing, "
            f"{len(extra)} unexpected, {duplicates} duplicate records"
        )


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required run artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Required run artifact is missing: {path}")
    rows = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSONL artifact {path}: {error}") from error
    if not rows:
        raise ValueError(f"Required run artifact is empty: {path}")
    return tuple(rows)


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"Expected object field {key!r}")
    return item


def _sequence(value: Mapping[str, Any], key: str) -> Sequence[Any]:
    item = value.get(key)
    if not isinstance(item, list) or not item:
        raise ValueError(f"Expected non-empty list field {key!r}")
    return item


def _text_sequence(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    items = _sequence(value, key)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"Expected text values in field {key!r}")
    return tuple(items)


def _text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"Expected non-empty text field {key!r}")
    return item


def _number(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"Expected numeric field {key!r}")
    return float(item)


def _mean(values: Iterable[float]) -> float:
    items = tuple(values)
    if not items:
        raise ValueError("Cannot average an empty sequence")
    return sum(items) / len(items)
