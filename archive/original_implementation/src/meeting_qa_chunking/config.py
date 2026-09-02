from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    tokenizer: str
    fixed_tokens: int
    turn_packed_tokens: int

    def __post_init__(self) -> None:
        if self.fixed_tokens <= 0 or self.turn_packed_tokens <= 0:
            raise ValueError("Chunk sizes must be positive")


@dataclass(frozen=True, slots=True)
class LumberChunkerConfig:
    window_tokens: int
    temperature: float
    max_attempts: int

    def __post_init__(self) -> None:
        if self.window_tokens <= 0:
            raise ValueError("LumberChunker window must be positive")
        if self.max_attempts <= 0:
            raise ValueError("LumberChunker max_attempts must be positive")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    model: str
    revision: str
    query_prefix: str
    max_sequence_tokens: int
    primary_budget: int
    evaluation_budgets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.revision:
            raise ValueError("Retrieval model revision cannot be empty")
        if self.max_sequence_tokens <= 0:
            raise ValueError("Retrieval max_sequence_tokens must be positive")
        if self.primary_budget not in self.evaluation_budgets:
            raise ValueError("primary_budget must be an evaluation budget")
        if any(budget <= 0 for budget in self.evaluation_budgets):
            raise ValueError("Retrieval budgets must be positive")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float


@dataclass(frozen=True, slots=True)
class APIConfig:
    base_url: str
    api_key_env: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("API base_url cannot be empty")
        if not self.api_key_env:
            raise ValueError("API api_key_env cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("API timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    max_completion_tokens: int
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Model name cannot be empty")
        if self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    boundary: ModelConfig
    answer: ModelConfig
    judge: ModelConfig


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    judge_temperature: float
    judge_max_attempts: int
    bootstrap_samples: int

    def __post_init__(self) -> None:
        if self.judge_max_attempts <= 0:
            raise ValueError("judge_max_attempts must be positive")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")


@dataclass(frozen=True, slots=True)
class Config:
    seed: int
    chunking: ChunkingConfig
    lumberchunker: LumberChunkerConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    api: APIConfig
    models: ModelsConfig

    def __post_init__(self) -> None:
        if self.chunking.tokenizer != self.retrieval.model:
            raise ValueError("Chunking tokenizer and retrieval model must match")


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as config_file:
        raw = tomllib.load(config_file)

    return Config(
        seed=int(raw["seed"]),
        chunking=ChunkingConfig(**raw["chunking"]),
        lumberchunker=LumberChunkerConfig(**raw["lumberchunker"]),
        retrieval=RetrievalConfig(
            model=raw["retrieval"]["model"],
            revision=raw["retrieval"]["revision"],
            query_prefix=raw["retrieval"]["query_prefix"],
            max_sequence_tokens=int(raw["retrieval"]["max_sequence_tokens"]),
            primary_budget=int(raw["retrieval"]["primary_budget"]),
            evaluation_budgets=tuple(raw["retrieval"]["evaluation_budgets"]),
        ),
        generation=GenerationConfig(**raw["generation"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        api=APIConfig(**raw["api"]),
        models=ModelsConfig(
            boundary=ModelConfig(**raw["models"]["boundary"]),
            answer=ModelConfig(**raw["models"]["answer"]),
            judge=ModelConfig(**raw["models"]["judge"]),
        ),
    )
