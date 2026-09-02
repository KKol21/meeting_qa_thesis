from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Callable, Literal, Sequence

from .chunking import LumberChunker, fixed_token_chunks, turn_packed_chunks
from .chunking.lumber import BoundaryDecision
from .config import Config
from .evaluation.elitr import (
    ELITRJudgment,
    judge_elitr_answer,
    parse_elitr_judgment,
)
from .evaluation.retrieval import TokenScores, score_evidence
from .evaluation.rouge import RougeScores, score_rouge
from .generation import generate_answer
from .models import TextModel
from .retrieval import DenseIndex, Embedder, Evidence, RankedChunk, project_evidence
from .schema import Chunk, DatasetSplit, Meeting, Query, Span
from .tokenization import TokenizedTranscript, Tokenizer, tokenize_meeting


ChunkMethod = Literal["fixed", "turn_packed", "lumber"]
_METHODS = {"fixed", "turn_packed", "lumber"}


@dataclass(frozen=True, slots=True)
class RunSettings:
    fixed_tokens: int = 256
    turn_packed_tokens: int = 256
    lumber_window_tokens: int = 550
    lumber_temperature: float = 0.1
    lumber_max_attempts: int = 2
    retrieval_budget: int = 512
    generation_temperature: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.fixed_tokens,
            self.turn_packed_tokens,
            self.lumber_window_tokens,
            self.lumber_max_attempts,
            self.retrieval_budget,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Token counts, attempts, and budget must be positive")

    @classmethod
    def from_config(cls, config: Config) -> RunSettings:
        return cls(
            fixed_tokens=config.chunking.fixed_tokens,
            turn_packed_tokens=config.chunking.turn_packed_tokens,
            lumber_window_tokens=config.lumberchunker.window_tokens,
            lumber_temperature=config.lumberchunker.temperature,
            lumber_max_attempts=config.lumberchunker.max_attempts,
            retrieval_budget=config.retrieval.primary_budget,
            generation_temperature=config.generation.temperature,
        )


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    meeting: Meeting
    method: ChunkMethod
    chunks: tuple[Chunk, ...]
    decisions: tuple[BoundaryDecision, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: Query
    method: ChunkMethod
    budget: int
    evidence: Evidence
    ranking: tuple[RankedChunk, ...]


@dataclass(frozen=True, slots=True)
class RunResult:
    retrieval: RetrievalResult
    answer: str

    @property
    def query(self) -> Query:
        return self.retrieval.query

    @property
    def method(self) -> ChunkMethod:
        return self.retrieval.method

    @property
    def budget(self) -> int:
        return self.retrieval.budget

    @property
    def evidence(self) -> Evidence:
        return self.retrieval.evidence


@dataclass(frozen=True, slots=True)
class QMSumEvaluation:
    result: RunResult
    retrieval: TokenScores
    rouge: RougeScores


@dataclass(frozen=True, slots=True)
class ELITREvaluation:
    result: RunResult
    judgment: ELITRJudgment


class JsonCache:
    """Auditable file cache. Change ``namespace`` when a model or config changes."""

    def __init__(self, root: str | Path, namespace: str) -> None:
        if not namespace.strip():
            raise ValueError("Cache namespace must not be empty")
        self.root = Path(root)
        self.namespace = namespace

    def read(self, stage: str, inputs: Any) -> Any | None:
        canonical_inputs = _json_value(inputs)
        path = self._path(stage, canonical_inputs)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as source:
            envelope = json.load(source)
        if (
            envelope.get("namespace") != self.namespace
            or envelope.get("inputs") != canonical_inputs
        ):
            raise ValueError(f"Cache metadata does not match {path}")
        return envelope["value"]

    def write(self, stage: str, inputs: Any, value: Any) -> None:
        canonical_inputs = _json_value(inputs)
        path = self._path(stage, canonical_inputs)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        envelope = {
            "namespace": self.namespace,
            "inputs": canonical_inputs,
            "value": value,
        }
        temporary.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def _path(self, stage: str, inputs: Any) -> Path:
        payload = json.dumps(
            {"namespace": self.namespace, "inputs": inputs},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        key = sha256(payload.encode("utf-8")).hexdigest()
        return self.root / stage / f"{key}.json"


class CachedTextModel:
    def __init__(
        self,
        model: TextModel,
        cache: JsonCache | None,
        stage: str,
        validator: Callable[[str, str], bool] | None = None,
    ) -> None:
        self.model = model
        self.cache = cache
        self.stage = stage
        self.validator = validator

    def complete(self, prompt: str, *, temperature: float) -> str:
        key = {"prompt": prompt, "temperature": temperature}
        cached = self.cache.read(self.stage, key) if self.cache else None
        if cached is not None:
            text = str(cached["text"])
            if self.validator is None or self.validator(prompt, text):
                return text
        text = self.model.complete(prompt, temperature=temperature)
        valid = self.validator is None or self.validator(prompt, text)
        if self.cache and valid:
            self.cache.write(self.stage, key, {"text": text})
        return text


class CachedEmbedder:
    def __init__(self, embedder: Embedder, cache: JsonCache | None) -> None:
        self.embedder = embedder
        self.cache = cache

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._embed("document_embeddings", texts, self.embedder.embed_documents)

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._embed("query_embeddings", texts, self.embedder.embed_queries)

    def _embed(
        self,
        stage: str,
        texts: Sequence[str],
        compute: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    ) -> tuple[tuple[float, ...], ...]:
        key = {"texts": list(texts)}
        cached = self.cache.read(stage, key) if self.cache else None
        if cached is None:
            vectors = tuple(tuple(float(value) for value in row) for row in compute(texts))
            if self.cache:
                self.cache.write(stage, key, {"vectors": vectors})
            return vectors
        return tuple(tuple(float(value) for value in row) for row in cached["vectors"])


class ExperimentRunner:
    """Run the same retrieval-and-answer pipeline for any supported dataset split."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        embedder: Embedder | None = None,
        answer_model: TextModel | None = None,
        *,
        boundary_model: TextModel | None = None,
        judge_model: TextModel | None = None,
        settings: RunSettings = RunSettings(),
        cache: JsonCache | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.embedder = CachedEmbedder(embedder, cache) if embedder else None
        self.answer_model = (
            CachedTextModel(answer_model, cache, "answers", _valid_answer)
            if answer_model is not None
            else None
        )
        self.boundary_model = (
            CachedTextModel(
                boundary_model,
                cache,
                "lumber_boundaries",
                _valid_lumber_boundary,
            )
            if boundary_model is not None
            else None
        )
        self.judge_model = (
            CachedTextModel(judge_model, cache, "elitr_judgments", _valid_elitr_judgment)
            if judge_model is not None
            else None
        )
        self.settings = settings
        self.cache = cache

    def run(
        self,
        dataset: DatasetSplit,
        methods: Sequence[ChunkMethod] = ("fixed", "turn_packed", "lumber"),
        *,
        budget: int | None = None,
    ) -> tuple[RunResult, ...]:
        """Convenience wrapper that retrieves and generates at one token budget."""

        selected_budget = self.settings.retrieval_budget if budget is None else budget
        return self.generate(
            self.retrieve(dataset, methods, budgets=(selected_budget,))
        )

    def segment(
        self,
        dataset: DatasetSplit,
        methods: Sequence[ChunkMethod] = ("fixed", "turn_packed", "lumber"),
    ) -> tuple[SegmentationResult, ...]:
        requested = self._validate_methods(methods)
        results: list[SegmentationResult] = []
        for meeting in dataset.meetings:
            transcript = tokenize_meeting(meeting, self.tokenizer)
            for method in requested:
                chunks, decisions = self._segmentation(transcript, method)
                results.append(SegmentationResult(meeting, method, chunks, decisions))
        return tuple(results)

    def retrieve(
        self,
        dataset: DatasetSplit,
        methods: Sequence[ChunkMethod] = ("fixed", "turn_packed", "lumber"),
        *,
        budgets: Sequence[int] | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Rank once per query and project that ranking at every requested budget."""

        requested = self._validate_methods(methods)
        if self.embedder is None:
            raise ValueError("Retrieval requires an embedder")
        requested_budgets = (
            (self.settings.retrieval_budget,) if budgets is None else tuple(budgets)
        )
        if not requested_budgets or any(budget <= 0 for budget in requested_budgets):
            raise ValueError("Retrieval budgets must be positive")
        if len(set(requested_budgets)) != len(requested_budgets):
            raise ValueError("Retrieval budgets must be unique")

        queries = _queries_by_meeting(dataset.queries)
        results: list[RetrievalResult] = []
        for meeting in dataset.meetings:
            transcript = tokenize_meeting(meeting, self.tokenizer)
            for method in requested:
                chunks = self._chunks(transcript, method)
                index = DenseIndex.build(chunks, self.embedder)
                for query in queries.get(meeting.id, ()):
                    ranking = index.search(query.text)
                    for budget in requested_budgets:
                        evidence = project_evidence(
                            ranking,
                            transcript,
                            self.tokenizer,
                            budget,
                        )
                        results.append(
                            RetrievalResult(query, method, budget, evidence, ranking)
                        )
        return tuple(results)

    def generate(
        self,
        retrievals: Sequence[RetrievalResult],
        *,
        budgets: Sequence[int] | None = None,
    ) -> tuple[RunResult, ...]:
        """Generate answers only for selected retrieval budgets."""

        if self.answer_model is None:
            raise ValueError("Answer generation requires an answer model")
        selected = None if budgets is None else set(budgets)
        if selected is not None:
            if not selected or any(budget <= 0 for budget in selected):
                raise ValueError("Generation budgets must be positive")
            missing = selected - {result.budget for result in retrievals}
            if missing:
                raise ValueError(f"No retrieval results for budgets: {sorted(missing)}")

        results: list[RunResult] = []
        for retrieval in retrievals:
            if selected is not None and retrieval.budget not in selected:
                continue
            answer = generate_answer(
                self.answer_model,
                retrieval.query,
                retrieval.evidence,
                temperature=self.settings.generation_temperature,
            )
            results.append(RunResult(retrieval, answer))
        return tuple(results)

    def evaluate_elitr(
        self,
        result: RunResult,
        *,
        temperature: float = 0.0,
        max_attempts: int = 2,
    ) -> ELITREvaluation:
        if result.query.dataset != "elitr":
            raise ValueError("ELITR evaluation requires an ELITR query")
        if self.judge_model is None:
            raise ValueError("ELITR evaluation requires a judge model")
        judgment = judge_elitr_answer(
            self.judge_model,
            result.query,
            result.answer,
            temperature=temperature,
            max_attempts=max_attempts,
        )
        return ELITREvaluation(result, judgment)

    def _chunks(
        self,
        transcript: TokenizedTranscript,
        method: ChunkMethod,
    ) -> tuple[Chunk, ...]:
        return self._segmentation(transcript, method)[0]

    def _segmentation(
        self,
        transcript: TokenizedTranscript,
        method: ChunkMethod,
    ) -> tuple[tuple[Chunk, ...], tuple[BoundaryDecision, ...]]:
        key = {
            "meeting": _meeting_dict(transcript.meeting),
            "method": method,
            "settings": asdict(self.settings),
        }
        cached = self.cache.read("chunks", key) if self.cache else None
        if cached is not None:
            return (
                tuple(_chunk_from_dict(item) for item in cached["chunks"]),
                tuple(
                    _decision_from_dict(item) for item in cached.get("decisions", ())
                ),
            )

        decisions: tuple[BoundaryDecision, ...] = ()
        if method == "fixed":
            chunks = fixed_token_chunks(
                transcript, self.tokenizer, self.settings.fixed_tokens
            )
        elif method == "turn_packed":
            chunks = turn_packed_chunks(transcript, self.settings.turn_packed_tokens)
        else:
            assert self.boundary_model is not None
            result = LumberChunker(
                self.boundary_model,
                window_tokens=self.settings.lumber_window_tokens,
                temperature=self.settings.lumber_temperature,
                max_attempts=self.settings.lumber_max_attempts,
            ).segment(transcript)
            chunks = result.chunks
            decisions = result.decisions

        if self.cache:
            self.cache.write(
                "chunks",
                key,
                {
                    "chunks": [_chunk_dict(chunk) for chunk in chunks],
                    "decisions": [_decision_dict(item) for item in decisions],
                },
            )
        return chunks, decisions

    def _validate_methods(
        self, methods: Sequence[ChunkMethod]
    ) -> tuple[ChunkMethod, ...]:
        requested = tuple(methods)
        unknown = set(requested) - _METHODS
        if unknown:
            raise ValueError(f"Unknown chunking methods: {sorted(unknown)}")
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Chunking methods must be non-empty and unique")
        if "lumber" in requested and self.boundary_model is None:
            raise ValueError("LumberChunker requires a boundary model")
        return requested


def evaluate_qmsum(
    result: RunResult,
    transcript: TokenizedTranscript,
    *,
    rouge_scorer: Callable[[str, str], RougeScores] = score_rouge,
) -> QMSumEvaluation:
    if result.query.dataset != "qmsum":
        raise ValueError("QMSum evaluation requires a QMSum query")
    retrieval = evaluate_qmsum_retrieval(result.retrieval, transcript)
    rouge = rouge_scorer(result.query.reference_answer, result.answer)
    return QMSumEvaluation(result, retrieval, rouge)


def evaluate_qmsum_retrieval(
    result: RetrievalResult,
    transcript: TokenizedTranscript,
) -> TokenScores:
    if result.query.dataset != "qmsum":
        raise ValueError("QMSum retrieval evaluation requires a QMSum query")
    return score_evidence(result.query, result.evidence, transcript)


def _queries_by_meeting(queries: Sequence[Query]) -> dict[str, tuple[Query, ...]]:
    grouped: dict[str, list[Query]] = {}
    for query in queries:
        grouped.setdefault(query.meeting_id, []).append(query)
    return {meeting_id: tuple(items) for meeting_id, items in grouped.items()}


def _meeting_dict(meeting: Meeting) -> dict[str, Any]:
    return {
        "dataset": meeting.dataset,
        "split": meeting.split,
        "id": meeting.id,
        "turns": [
            {"id": turn.id, "speaker": turn.speaker, "text": turn.text}
            for turn in meeting.turns
        ],
    }


def _chunk_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "meeting_id": chunk.meeting_id,
        "method": chunk.method,
        "text": chunk.text,
        "source_spans": [[span.start, span.end] for span in chunk.source_spans],
        "turn_ids": list(chunk.turn_ids),
    }


def _chunk_from_dict(raw: dict[str, Any]) -> Chunk:
    return Chunk(
        id=str(raw["id"]),
        meeting_id=str(raw["meeting_id"]),
        method=str(raw["method"]),
        text=str(raw["text"]),
        source_spans=tuple(Span(int(start), int(end)) for start, end in raw["source_spans"]),
        turn_ids=tuple(int(turn_id) for turn_id in raw["turn_ids"]),
    )


def _decision_dict(decision: BoundaryDecision) -> dict[str, Any]:
    return {
        "window_turn_ids": list(decision.window_turn_ids),
        "boundary_turn_id": decision.boundary_turn_id,
        "response": decision.response,
        "prompt": decision.prompt,
    }


def _decision_from_dict(raw: dict[str, Any]) -> BoundaryDecision:
    return BoundaryDecision(
        window_turn_ids=tuple(int(value) for value in raw["window_turn_ids"]),
        boundary_turn_id=int(raw["boundary_turn_id"]),
        response=str(raw["response"]),
        prompt=str(raw["prompt"]),
    )


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _valid_answer(_prompt: str, response: str) -> bool:
    return bool(response.strip())


def _valid_lumber_boundary(prompt: str, response: str) -> bool:
    ids = re.findall(r"^ID (\d+):", prompt, flags=re.MULTILINE)
    answer = re.search(r"Answer:\s*ID\s*:?[ ]*(\d+)", response, re.IGNORECASE)
    return answer is not None and int(answer.group(1)) in {int(value) for value in ids[1:]}


def _valid_elitr_judgment(_prompt: str, response: str) -> bool:
    try:
        parse_elitr_judgment(response)
    except ValueError:
        return False
    return True
