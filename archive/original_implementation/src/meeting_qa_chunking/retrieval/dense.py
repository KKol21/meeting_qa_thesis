from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Protocol, Sequence

from ..schema import Chunk


Vector = Sequence[float]


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]: ...

    def embed_queries(self, texts: Sequence[str]) -> Sequence[Vector]: ...


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk: Chunk
    score: float
    rank: int


def _normalize(vector: Vector) -> tuple[float, ...]:
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("Embedding vectors must be non-zero")
    return tuple(value / norm for value in vector)


class DenseIndex:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        embedder: Embedder,
    ) -> None:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must be non-empty and aligned")

        self.chunks = tuple(chunks)
        self.vectors = tuple(_normalize(vector) for vector in vectors)
        if len({len(vector) for vector in self.vectors}) != 1:
            raise ValueError("Embedding vectors must have one shared dimension")
        self.embedder = embedder

    @classmethod
    def build(cls, chunks: Sequence[Chunk], embedder: Embedder) -> DenseIndex:
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        return cls(chunks, vectors, embedder)

    def search(self, query: str) -> tuple[RankedChunk, ...]:
        query_vectors = self.embedder.embed_queries([query])
        if len(query_vectors) != 1:
            raise ValueError("Embedder must return exactly one query vector")
        query_vector = _normalize(query_vectors[0])
        if len(query_vector) != len(self.vectors[0]):
            raise ValueError("Query and document embeddings have different dimensions")

        scored = [
            (sum(left * right for left, right in zip(query_vector, vector)), index)
            for index, vector in enumerate(self.vectors)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(
            RankedChunk(self.chunks[index], score, rank)
            for rank, (score, index) in enumerate(scored, start=1)
        )
