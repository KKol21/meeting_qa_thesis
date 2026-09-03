"""Typed experiment presets and pinned model revisions."""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    name: str
    revision: str
    prequantized: bool = False


LUMBER_MODEL = ModelSpec(
    tag="qwen2.5-7b",
    name="Qwen/Qwen2.5-7B-Instruct",
    revision="a09a35458c702b33eeacc393d103063234e8bc28",
)
QWEN_14B = ModelSpec(
    tag="qwen2.5-14b",
    name="Qwen/Qwen2.5-14B-Instruct",
    revision="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
)
QWEN_32B_BNB4 = ModelSpec(
    tag="qwen2.5-32b-bnb4",
    name="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    revision="aa79e3472818bdec779075d80928602591d9f2a0",
    prequantized=True,
)
DENSE_RETRIEVER_MODEL = ModelSpec(
    tag="gte-modernbert-base",
    name="Alibaba-NLP/gte-modernbert-base",
    revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
)
JUDGE_MODEL = ModelSpec(
    tag="llama-3.3-70b-bnb4",
    name="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    revision="74be54198eaf4f3c7fba1f4e9fa63725a810c7eb",
    prequantized=True,
)
BERTSCORE_MODEL = ModelSpec(
    tag="roberta-large",
    name="FacebookAI/roberta-large",
    revision="722cf37b1afa9454edce342e7895e588b6ff1d59",
)

ANSWER_MODELS = {
    model.tag: model for model in (LUMBER_MODEL, QWEN_14B, QWEN_32B_BNB4)
}
SEGMENTATION_MODELS = {LUMBER_MODEL.tag: LUMBER_MODEL}
DENSE_MODELS = {DENSE_RETRIEVER_MODEL.tag: DENSE_RETRIEVER_MODEL}
JUDGE_MODELS = {JUDGE_MODEL.tag: JUDGE_MODEL}
BERTSCORE_MODELS = {BERTSCORE_MODEL.tag: BERTSCORE_MODEL}

CHUNKERS = ("turn_packed", "word_packed", "lumber")
RETRIEVERS = ("dense", "bm25", "hybrid")
EVIDENCE_ORDERS = ("ranked", "chronological")


@dataclass(frozen=True)
class ConditionSpec:
    chunker: str
    retriever: str
    evidence_words: int

    @property
    def name(self) -> str:
        return f"{self.chunker}__{self.retriever}__w{self.evidence_words}"

    def to_dict(self) -> dict[str, object]:
        return {
            "chunker": self.chunker,
            "retriever": self.retriever,
            "evidence_words": self.evidence_words,
        }


def retrieval_conditions(
    evidence_budgets: tuple[int, ...] | list[int],
    chunkers: tuple[str, ...] = CHUNKERS,
    retrievers: tuple[str, ...] = RETRIEVERS,
) -> tuple[ConditionSpec, ...]:
    if not evidence_budgets or any(words <= 0 for words in evidence_budgets):
        raise ValueError("Evidence budgets must be positive")
    return tuple(
        ConditionSpec(chunker, retriever, words)
        for chunker in chunkers
        for retriever in retrievers
        for words in evidence_budgets
    )


@dataclass(frozen=True)
class SegmentationSpec:
    model: ModelSpec
    target_tokens: int
    max_new_tokens: int
    temperature: float
    seed: int


@dataclass(frozen=True)
class RetrievalSpec:
    dense_model: ModelSpec
    chunkers: tuple[str, ...]
    retrievers: tuple[str, ...]
    turn_packed_max_words: int
    word_packed_max_words: int
    evidence_budgets: tuple[int, ...]
    evidence_order: str
    bm25_k1: float
    bm25_b: float
    rrf_k: int


@dataclass(frozen=True)
class GenerationSpec:
    max_new_tokens: int
    temperature: float
    seed: int


@dataclass(frozen=True)
class AnswerStageSpec:
    name: str
    source: str
    model: ModelSpec


@dataclass(frozen=True)
class EvaluationSpec:
    bertscore_model: ModelSpec
    bertscore_layers: int
    bertscore_batch_size: int
    judge_model: ModelSpec
    judge_max_new_tokens: int
    judge_temperature: float
    judge_seed: int


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_root: Path
    data_dir: Path
    lumber_dir: Path
    meetings: tuple[str, ...]
    meeting_manifest: Path | None
    run_segmentation: bool
    segmentation: SegmentationSpec
    retrieval: RetrievalSpec
    generation: GenerationSpec
    answers: tuple[AnswerStageSpec, ...]
    run_evaluation: bool
    evaluation: EvaluationSpec

    @property
    def retrieval_dir(self) -> Path:
        return self.output_root / "retrieval"

    @property
    def answers_dir(self) -> Path:
        return self.output_root / "answers"

    @property
    def evaluation_dir(self) -> Path:
        return self.output_root / "evaluation"

    def answer_stage(self, name: str) -> AnswerStageSpec:
        for stage in self.answers:
            if stage.name == name:
                return stage
        raise ValueError(f"Unknown answer stage: {name}")

    def meeting_ids(self, repository_root: Path = Path(".")) -> list[str]:
        if self.meetings:
            return list(self.meetings)
        if self.meeting_manifest is None:
            raise ValueError("Run config has no meeting selection")
        path = repository_root / self.meeting_manifest
        lines = path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]


def _model(
    section: dict[str, object],
    key: str,
    registry: dict[str, ModelSpec],
) -> ModelSpec:
    tag = section[key]
    if tag not in registry:
        raise ValueError(f"Unknown model for {key}: {tag}")
    return registry[tag]


def _positive(value: object, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _temperature(value: object, name: str) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def load_run_config(path: Path) -> RunConfig:
    """Load the single preset that controls a complete experiment run."""

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    run = raw["run"]
    segmentation = raw["segmentation"]
    retrieval = raw["retrieval"]
    generation = raw["generation"]
    evaluation = raw["evaluation"]

    meetings = tuple(run.get("meetings", ()))
    manifest_value = run.get("meeting_manifest")
    manifest = Path(manifest_value) if manifest_value else None
    if bool(meetings) == bool(manifest):
        raise ValueError("Specify exactly one of meetings or meeting_manifest")

    budgets = tuple(
        _positive(value, "evidence budget")
        for value in retrieval["evidence_budgets"]
    )
    chunkers = tuple(retrieval["chunkers"])
    retrievers = tuple(retrieval["retrievers"])
    if (
        not chunkers
        or len(set(chunkers)) != len(chunkers)
        or any(name not in CHUNKERS for name in chunkers)
    ):
        raise ValueError(f"Unknown chunker list: {chunkers}")
    if (
        not retrievers
        or len(set(retrievers)) != len(retrievers)
        or any(name not in RETRIEVERS for name in retrievers)
    ):
        raise ValueError(f"Unknown retriever list: {retrievers}")
    retrieval_conditions(budgets, chunkers, retrievers)
    evidence_order = retrieval["evidence_order"]
    if evidence_order not in EVIDENCE_ORDERS:
        raise ValueError(f"Unknown evidence order: {evidence_order}")

    answer_stages = []
    for item in raw.get("answers", []):
        source = item["source"]
        if source not in ("oracle", "retrieval"):
            raise ValueError(f"Unknown answer source: {source}")
        answer_stages.append(
            AnswerStageSpec(
                name=item["name"],
                source=source,
                model=_model(item, "model", ANSWER_MODELS),
            )
        )
    if len({stage.name for stage in answer_stages}) != len(answer_stages):
        raise ValueError("Answer stage names must be unique")
    if not answer_stages:
        raise ValueError("At least one answer stage is required")

    return RunConfig(
        name=run["name"],
        output_root=Path(run["output_root"]),
        data_dir=Path(run["data_dir"]),
        lumber_dir=Path(run["lumber_dir"]),
        meetings=meetings,
        meeting_manifest=manifest,
        run_segmentation=bool(run.get("run_segmentation", True)),
        segmentation=SegmentationSpec(
            model=_model(segmentation, "model", SEGMENTATION_MODELS),
            target_tokens=_positive(
                segmentation["target_tokens"], "target_tokens"
            ),
            max_new_tokens=_positive(
                segmentation["max_new_tokens"], "segmentation max_new_tokens"
            ),
            temperature=_temperature(
                segmentation["temperature"], "segmentation temperature"
            ),
            seed=int(segmentation["seed"]),
        ),
        retrieval=RetrievalSpec(
            dense_model=_model(retrieval, "dense_model", DENSE_MODELS),
            chunkers=chunkers,
            retrievers=retrievers,
            turn_packed_max_words=_positive(
                retrieval["turn_packed_max_words"],
                "turn_packed_max_words",
            ),
            word_packed_max_words=_positive(
                retrieval["word_packed_max_words"],
                "word_packed_max_words",
            ),
            evidence_budgets=budgets,
            evidence_order=evidence_order,
            bm25_k1=float(retrieval["bm25_k1"]),
            bm25_b=float(retrieval["bm25_b"]),
            rrf_k=_positive(retrieval["rrf_k"], "rrf_k"),
        ),
        generation=GenerationSpec(
            max_new_tokens=_positive(
                generation["max_new_tokens"], "answer max_new_tokens"
            ),
            temperature=_temperature(
                generation["temperature"], "answer temperature"
            ),
            seed=int(generation["seed"]),
        ),
        answers=tuple(answer_stages),
        run_evaluation=bool(run.get("run_evaluation", True)),
        evaluation=EvaluationSpec(
            bertscore_model=_model(
                evaluation, "bertscore_model", BERTSCORE_MODELS
            ),
            bertscore_layers=_positive(
                evaluation["bertscore_layers"], "bertscore_layers"
            ),
            bertscore_batch_size=_positive(
                evaluation["bertscore_batch_size"], "bertscore_batch_size"
            ),
            judge_model=_model(evaluation, "judge_model", JUDGE_MODELS),
            judge_max_new_tokens=_positive(
                evaluation["judge_max_new_tokens"], "judge_max_new_tokens"
            ),
            judge_temperature=_temperature(
                evaluation["judge_temperature"], "judge temperature"
            ),
            judge_seed=int(evaluation["judge_seed"]),
        ),
    )
