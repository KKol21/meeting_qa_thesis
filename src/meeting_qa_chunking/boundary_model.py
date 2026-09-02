"""Local model configuration for LumberChunker boundaries."""

from pathlib import Path

from .local_model import LocalChatModel


class LocalBoundaryChooser(LocalChatModel):
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        max_new_tokens: int = 32,
        seed: int = 42,
        temperature: float = 0.1,
        cache_dir: Path = Path(".cache/lumber"),
    ) -> None:
        super().__init__(
            model_name=model_name,
            revision=revision,
            max_new_tokens=max_new_tokens,
            seed=seed,
            temperature=temperature,
            cache_dir=cache_dir,
        )
