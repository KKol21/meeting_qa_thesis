from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Sequence

from .artifacts import append_jsonl, write_jsonl
from .config import Config, load_config
from .data import load_elitr, load_qmsum
from .model_backends import OpenAICompatibleModel
from .models import CallBudget, LimitedTextModel
from .pipeline import (
    ChunkMethod,
    ExperimentRunner,
    JsonCache,
    RetrievalResult,
    RunSettings,
    SegmentationResult,
    evaluate_qmsum,
    evaluate_qmsum_retrieval,
)
from .provenance import build_manifest, ensure_manifest
from .retrieval import SentenceTransformerEmbedder
from .schema import DatasetSplit
from .summarize import summarize_run
from .tokenization import HuggingFaceTokenizer, TokenizedTranscript, tokenize_meeting


_EXPECTED_COUNTS = {
    "qmsum-val": (35, 237),
    "qmsum-test": (35, 244),
    "elitr-dev": (10, 141),
    "elitr-test2": (8, 130),
}
_METHODS: tuple[ChunkMethod, ...] = ("fixed", "turn_packed", "lumber")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meeting-qa")
    parser.add_argument("--config", default="configs/baseline.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-config")

    validate_data = commands.add_parser("validate-data")
    validate_data.add_argument("--qmsum-root", default="data/raw/qmsum/data/ALL")
    validate_data.add_argument("--elitr-root", default="data/raw/elitr-bench")

    summarize = commands.add_parser("summarize")
    summarize.add_argument("--run-dir", required=True)

    run = commands.add_parser("run")
    run.add_argument("--dataset", choices=("qmsum", "elitr"), required=True)
    run.add_argument("--split", required=True)
    run.add_argument(
        "--stage",
        choices=("segment", "retrieve", "answer", "evaluate", "all"),
        default="all",
    )
    run.add_argument("--methods", nargs="+", choices=_METHODS, default=_METHODS)
    run.add_argument("--budgets", nargs="+", type=_positive_int)
    run.add_argument("--answer-budget", type=_positive_int)
    run.add_argument("--run-dir", default="runs/baseline")
    run.add_argument("--qmsum-root", default="data/raw/qmsum/data/ALL")
    run.add_argument("--elitr-root", default="data/raw/elitr-bench")
    run.add_argument("--meeting-id")
    run.add_argument("--query-id")
    run.add_argument("--limit-meetings", type=_positive_int)
    run.add_argument("--limit-queries", type=_positive_int)
    run.add_argument("--max-api-calls", type=_positive_int, default=20)
    run.add_argument("--dry-run", action="store_true")
    return parser


def _validate_data(qmsum_root: str, elitr_root: str) -> None:
    qmsum = Path(qmsum_root)
    elitr = Path(elitr_root)
    corpus = elitr / "elitr-minuting-corpus-en"
    datasets = {
        "qmsum-val": load_qmsum(qmsum / "val", "val"),
        "qmsum-test": load_qmsum(qmsum / "test", "test"),
        "elitr-dev": load_elitr(elitr / "data/elitr-bench-qa_dev.json", corpus / "dev"),
        "elitr-test2": load_elitr(
            elitr / "data/elitr-bench-qa_test2.json", corpus / "test2"
        ),
    }
    for name, dataset in datasets.items():
        actual = _require_official_counts(name, dataset)
        print(f"{name}: {actual[0]} meetings, {actual[1]} queries")


def _require_official_counts(
    name: str, dataset: DatasetSplit
) -> tuple[int, int]:
    actual = (len(dataset.meetings), len(dataset.queries))
    expected = _EXPECTED_COUNTS[name]
    if actual != expected:
        raise ValueError(f"{name}: expected {expected}, found {actual}")
    return actual


def _load_dataset(args: argparse.Namespace) -> tuple[DatasetSplit, tuple[Path, ...]]:
    if args.dataset == "qmsum":
        if args.split not in {"val", "test"}:
            raise ValueError("QMSum split must be val or test")
        source = Path(args.qmsum_root) / args.split
        return load_qmsum(source, args.split), (source,)

    if args.split not in {"dev", "test2"}:
        raise ValueError("ELITR split must be dev or test2")
    root = Path(args.elitr_root)
    benchmark = root / f"data/elitr-bench-qa_{args.split}.json"
    transcripts = root / "elitr-minuting-corpus-en" / args.split
    return load_elitr(benchmark, transcripts), (benchmark, transcripts)


def _select_dataset(dataset: DatasetSplit, args: argparse.Namespace) -> DatasetSplit:
    meetings = dataset.meetings
    queries = dataset.queries

    if args.query_id:
        matches = [query for query in queries if query.id == args.query_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one query {args.query_id!r}, found {len(matches)}")
        if args.meeting_id and matches[0].meeting_id != args.meeting_id:
            raise ValueError("query-id does not belong to meeting-id")
        meetings = tuple(
            meeting for meeting in meetings if meeting.id == matches[0].meeting_id
        )
        queries = (matches[0],)

    if args.meeting_id:
        meetings = tuple(meeting for meeting in meetings if meeting.id == args.meeting_id)
        if not meetings:
            raise ValueError(f"Meeting {args.meeting_id!r} was not found")
    if args.limit_meetings:
        meetings = meetings[: args.limit_meetings]

    meeting_ids = {meeting.id for meeting in meetings}
    queries = tuple(query for query in queries if query.meeting_id in meeting_ids)
    if args.limit_queries:
        queries = queries[: args.limit_queries]
    if not queries:
        raise ValueError("Selection contains no queries")
    selected_meetings = {query.meeting_id for query in queries}
    meetings = tuple(meeting for meeting in meetings if meeting.id in selected_meetings)
    return DatasetSplit(tuple(meetings), queries)


def _selection(
    args: argparse.Namespace,
    dataset: DatasetSplit,
    methods: Sequence[ChunkMethod],
    budgets: Sequence[int],
    answer_budget: int,
) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "split": args.split,
        "meeting_ids": [meeting.id for meeting in dataset.meetings],
        "query_ids": [query.id for query in dataset.queries],
        "methods": list(methods),
        "retrieval_budgets": list(budgets),
        "answer_budget": answer_budget,
    }


def _dry_run(
    args: argparse.Namespace,
    dataset: DatasetSplit,
    methods: Sequence[ChunkMethod],
) -> None:
    queries = len(dataset.queries)
    answer_calls = (
        queries * len(methods)
        if args.stage in {"answer", "evaluate", "all"}
        else 0
    )
    judge_calls = (
        answer_calls
        if args.dataset == "elitr" and args.stage in {"evaluate", "all"}
        else 0
    )
    print(f"{len(dataset.meetings)} meetings, {queries} queries, {len(methods)} methods")
    print(f"answer calls: {answer_calls}; ELITR judge calls: {judge_calls}")
    print(f"uncached API calls are capped at {args.max_api_calls} per invocation")
    if "lumber" in methods:
        print("Lumber boundary calls are data-dependent and cached once per local window.")


def _retrieval_record(
    result: RetrievalResult, transcript: TokenizedTranscript
) -> dict[str, Any]:
    record = {
        "query": asdict(result.query),
        "method": result.method,
        "budget": result.budget,
        "evidence": asdict(result.evidence),
        "ranking": [
            {"chunk_id": item.chunk.id, "score": item.score, "rank": item.rank}
            for item in result.ranking
        ],
    }
    if result.query.dataset == "qmsum":
        scores = evaluate_qmsum_retrieval(result, transcript)
        record["metrics"] = {**asdict(scores), "zero_hit": scores.zero_hit}
    return record


def _segmentation_record(result: SegmentationResult) -> dict[str, Any]:
    sizes = sorted(chunk.source_token_count for chunk in result.chunks)
    p90 = sizes[ceil(0.9 * len(sizes)) - 1]
    middle = (sizes[(len(sizes) - 1) // 2] + sizes[len(sizes) // 2]) / 2
    return {
        "dataset": result.meeting.dataset,
        "split": result.meeting.split,
        "meeting_id": result.meeting.id,
        "method": result.method,
        "chunks": [asdict(chunk) for chunk in result.chunks],
        "lumber_decisions": [asdict(item) for item in result.decisions],
        "diagnostics": {
            "chunk_count": len(sizes),
            "median_source_tokens": middle,
            "p90_source_tokens": p90,
            "max_source_tokens": sizes[-1],
        },
    }


def _run_experiment(args: argparse.Namespace, config: Config) -> None:
    original, data_paths = _load_dataset(args)
    _require_official_counts(f"{args.dataset}-{args.split}", original)
    dataset = _select_dataset(original, args)
    methods = tuple(args.methods)
    budgets = tuple(
        args.budgets
        or (
            config.retrieval.evaluation_budgets
            if args.dataset == "qmsum"
            else (config.retrieval.primary_budget,)
        )
    )
    answer_budget = args.answer_budget or config.retrieval.primary_budget
    if args.stage in {"answer", "evaluate", "all"} and answer_budget not in budgets:
        raise ValueError("answer-budget must also be listed in retrieval budgets")
    if args.dry_run:
        _dry_run(args, dataset, methods)
        return

    selection = _selection(args, dataset, methods, budgets, answer_budget)
    manifest = build_manifest(
        config,
        args.config,
        data_paths=data_paths,
        selection=selection,
    )
    run_dir = Path(args.run_dir)
    ensure_manifest(run_dir, manifest)
    cache = JsonCache(run_dir / "cache", manifest["fingerprint"])
    call_budget = CallBudget(args.max_api_calls)

    def text_model(model_config: Any, stage: str) -> LimitedTextModel:
        def observe(response: dict[str, Any]) -> None:
            append_jsonl(
                run_dir / "calls.jsonl",
                {
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "stage": stage,
                    **response,
                },
            )

        return LimitedTextModel(
            OpenAICompatibleModel(
                config.api, model_config, response_observer=observe
            ),
            call_budget,
        )

    tokenizer = HuggingFaceTokenizer(
        config.chunking.tokenizer, revision=config.retrieval.revision
    )
    needs_retrieval = args.stage in {"retrieve", "answer", "evaluate", "all"}
    embedder = (
        SentenceTransformerEmbedder(
            config.retrieval.model,
            revision=config.retrieval.revision,
            query_prefix=config.retrieval.query_prefix,
            max_sequence_tokens=config.retrieval.max_sequence_tokens,
        )
        if needs_retrieval
        else None
    )
    boundary_model = (
        text_model(config.models.boundary, "lumber_boundary")
        if "lumber" in methods
        else None
    )
    answer_model = (
        text_model(config.models.answer, "answer")
        if args.stage in {"answer", "evaluate", "all"}
        else None
    )
    judge_model = (
        text_model(config.models.judge, "elitr_judge")
        if args.dataset == "elitr" and args.stage in {"evaluate", "all"}
        else None
    )
    runner = ExperimentRunner(
        tokenizer,
        embedder,
        answer_model,
        boundary_model=boundary_model,
        judge_model=judge_model,
        settings=RunSettings.from_config(config),
        cache=cache,
    )

    if args.stage in {"segment", "all"}:
        segments = runner.segment(dataset, methods)
        write_jsonl(
            run_dir / "segments.jsonl",
            (_segmentation_record(result) for result in segments),
        )
        print(f"wrote {len(segments)} meeting-method segmentations")

    if not needs_retrieval:
        print(f"uncached API calls: {call_budget.used}")
        return
    retrievals = runner.retrieve(dataset, methods, budgets=budgets)
    transcripts = {
        meeting.id: tokenize_meeting(meeting, tokenizer) for meeting in dataset.meetings
    }
    write_jsonl(
        run_dir / "retrieval.jsonl",
        (
            _retrieval_record(result, transcripts[result.query.meeting_id])
            for result in retrievals
        ),
    )
    print(f"wrote {len(retrievals)} retrieval records")

    if args.stage == "retrieve":
        print(f"uncached API calls: {call_budget.used}")
        return
    answers = runner.generate(retrievals, budgets=(answer_budget,))
    write_jsonl(
        run_dir / "answers.jsonl",
        (
            {
                "query": asdict(result.query),
                "method": result.method,
                "budget": result.budget,
                "evidence": asdict(result.evidence),
                "answer": result.answer,
                "model": config.models.answer.name,
                "prompt_hash": manifest["prompts"]["answer"],
            }
            for result in answers
        ),
    )
    print(f"wrote {len(answers)} answers")

    if args.stage == "answer":
        print(f"uncached API calls: {call_budget.used}")
        return
    if args.dataset == "qmsum":
        evaluations = [
            evaluate_qmsum(result, transcripts[result.query.meeting_id])
            for result in answers
        ]
        write_jsonl(
            run_dir / "metrics.jsonl",
            (
                {
                    "dataset": "qmsum",
                    "query_id": item.result.query.id,
                    "meeting_id": item.result.query.meeting_id,
                    "method": item.result.method,
                    "budget": item.result.budget,
                    "retrieval": asdict(item.retrieval),
                    "rouge": asdict(item.rouge),
                }
                for item in evaluations
            ),
        )
        print(f"wrote {len(evaluations)} QMSum evaluations")
        print(f"uncached API calls: {call_budget.used}")
        return

    evaluations = [
        runner.evaluate_elitr(
            result,
            temperature=config.evaluation.judge_temperature,
            max_attempts=config.evaluation.judge_max_attempts,
        )
        for result in answers
    ]
    write_jsonl(
        run_dir / "judgments.jsonl",
        (
            {
                "query_id": item.result.query.id,
                "meeting_id": item.result.query.meeting_id,
                "method": item.result.method,
                "budget": item.result.budget,
                "answer": item.result.answer,
                "judgment": asdict(item.judgment),
            }
            for item in evaluations
        ),
    )
    write_jsonl(
        run_dir / "metrics.jsonl",
        (
            {
                "dataset": "elitr",
                "query_id": item.result.query.id,
                "meeting_id": item.result.query.meeting_id,
                "method": item.result.method,
                "budget": item.result.budget,
                "score": item.judgment.score,
                "question_type": item.result.query.question_type,
                "answer_position": item.result.query.answer_position,
            }
            for item in evaluations
        ),
    )
    print(f"wrote {len(evaluations)} ELITR judgments")
    print(f"uncached API calls: {call_budget.used}")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-config":
        config = load_config(args.config)
        print(f"Configuration is valid (seed={config.seed}).")
    elif args.command == "validate-data":
        _validate_data(args.qmsum_root, args.elitr_root)
    elif args.command == "summarize":
        summary = summarize_run(args.run_dir)
        print(
            f"wrote {Path(args.run_dir) / 'summary.json'} "
            f"for {summary['dataset']}"
        )
    elif args.command == "run":
        _run_experiment(args, load_config(args.config))


if __name__ == "__main__":
    main()
