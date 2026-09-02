"""Typed experiment settings shared by local and cluster entry points."""

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
DENSE_RETRIEVER_MODEL = ModelSpec(
    tag="gte-modernbert-base",
    name="Alibaba-NLP/gte-modernbert-base",
    revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
)
ANSWER_MODELS = {
    model.tag: model
    for model in (
        LUMBER_MODEL,
        ModelSpec(
            tag="qwen2.5-14b",
            name="Qwen/Qwen2.5-14B-Instruct",
            revision="cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        ),
        ModelSpec(
            tag="qwen2.5-32b-bnb4",
            name="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
            revision="aa79e3472818bdec779075d80928602591d9f2a0",
            prequantized=True,
        ),
    )
}
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

CHUNKERS = ("fixed", "lumber")
RETRIEVERS = ("dense", "bm25", "hybrid")


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
    evidence_budgets: list[int] | tuple[int, ...],
) -> tuple[ConditionSpec, ...]:
    if not evidence_budgets or any(words <= 0 for words in evidence_budgets):
        raise ValueError("Evidence budgets must be positive")
    return tuple(
        ConditionSpec(chunker, retriever, words)
        for chunker in CHUNKERS
        for retriever in RETRIEVERS
        for words in evidence_budgets
    )


@dataclass(frozen=True)
class AnswerStageSpec:
    name: str
    source: str
    model: ModelSpec


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_root: Path
    meetings: tuple[str, ...]
    meeting_manifest: Path | None
    run_segmentation: bool
    fixed_chunk_words: int
    evidence_budgets: tuple[int, ...]
    answers: tuple[AnswerStageSpec, ...]
    run_evaluation: bool

    def meeting_ids(self, repository_root: Path = Path(".")) -> list[str]:
        if self.meetings:
            return list(self.meetings)
        if self.meeting_manifest is None:
            raise ValueError("Run config has no meeting selection")
        path = repository_root / self.meeting_manifest
        lines = path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]


def load_run_config(path: Path) -> RunConfig:
    """Load and validate one small TOML experiment preset."""

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    run = raw.get("run", {})
    retrieval = raw.get("retrieval", {})
    meetings = tuple(run.get("meetings", ()))
    manifest_value = run.get("meeting_manifest")
    manifest = Path(manifest_value) if manifest_value else None
    if bool(meetings) == bool(manifest):
        raise ValueError("Specify exactly one of meetings or meeting_manifest")

    answers = []
    for item in raw.get("answers", []):
        model_tag = item["model"]
        if model_tag not in ANSWER_MODELS:
            raise ValueError(f"Unknown answer model: {model_tag}")
        source = item["source"]
        if source not in ("oracle", "retrieval"):
            raise ValueError(f"Unknown answer source: {source}")
        answers.append(
            AnswerStageSpec(item["name"], source, ANSWER_MODELS[model_tag])
        )

    fixed_chunk_words = int(retrieval.get("fixed_chunk_words", 256))
    budgets = tuple(int(value) for value in retrieval.get("evidence_budgets", (512, 1024)))
    if fixed_chunk_words <= 0:
        raise ValueError("fixed_chunk_words must be positive")
    retrieval_conditions(budgets)

    return RunConfig(
        name=run["name"],
        output_root=Path(run["output_root"]),
        meetings=meetings,
        meeting_manifest=manifest,
        run_segmentation=bool(run.get("run_segmentation", True)),
        fixed_chunk_words=fixed_chunk_words,
        evidence_budgets=budgets,
        answers=tuple(answers),
        run_evaluation=bool(run.get("run_evaluation", True)),
    )
