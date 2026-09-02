from __future__ import annotations

from typing import Sequence

from .dense import Vector


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        query_prefix: str = "",
        batch_size: int = 32,
        max_sequence_tokens: int | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install the experiment dependencies: pip install -e '.[experiment]'"
            ) from error

        try:
            self.model = SentenceTransformer(
                model_name, revision=revision, local_files_only=True
            )
        except OSError:
            self.model = SentenceTransformer(model_name, revision=revision)
        supported = int(self.model.max_seq_length)
        if max_sequence_tokens is not None:
            if not 0 < max_sequence_tokens <= supported:
                raise ValueError(
                    f"Requested max sequence length {max_sequence_tokens}; "
                    f"{model_name} supports {supported}"
                )
            self.model.max_seq_length = max_sequence_tokens
        self.max_sequence_tokens = int(self.model.max_seq_length)
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    def _encode(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        encoded = self.model.tokenizer(
            list(texts),
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        longest = max((len(token_ids) for token_ids in encoded), default=0)
        if longest > self.max_sequence_tokens:
            raise ValueError(
                f"Embedding input has {longest} tokens, exceeding the configured "
                f"limit of {self.max_sequence_tokens}; refusing silent truncation"
            )
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return tuple(tuple(float(value) for value in vector) for vector in vectors)

    def embed_documents(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        return self._encode(texts)

    def embed_queries(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        return self._encode([f"{self.query_prefix}{text}" for text in texts])
