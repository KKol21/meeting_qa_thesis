"""Rank meeting chunks with a local embedding model."""

import hashlib
import json
import math
from pathlib import Path
import re
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import Chunk
from .config import DENSE_RETRIEVER_MODEL


MODEL_NAME = DENSE_RETRIEVER_MODEL.name
MODEL_REVISION = DENSE_RETRIEVER_MODEL.revision
DEFAULT_CACHE_DIR = Path(".cache/embeddings")
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
TOKEN_PATTERN = re.compile(r"\w+")


def load_model(
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
) -> SentenceTransformer:
    try:
        return SentenceTransformer(
            model_name,
            revision=revision,
            local_files_only=True,
        )
    except OSError:
        return SentenceTransformer(model_name, revision=revision)


def cache_path(
    chunks: list[Chunk],
    cache_dir: Path,
    model_name: str,
    revision: str,
) -> Path:
    content = json.dumps(
        {
            "model": model_name,
            "revision": revision,
            "chunks": [chunk.text for chunk in chunks],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    return cache_dir / f"{hashlib.sha256(content).hexdigest()}.npy"


def load_or_encode_chunks(
    chunks: list[Chunk],
    model: SentenceTransformer,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
) -> tuple[np.ndarray, bool]:
    path = cache_path(chunks, cache_dir, model_name, revision)
    if path.exists():
        return np.load(path, allow_pickle=False), True

    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.npy")
    np.save(temporary_path, embeddings)
    temporary_path.replace(path)
    return embeddings, False


def rank_chunks(
    question: str,
    chunks: list[Chunk],
    model: SentenceTransformer,
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
) -> tuple[list[tuple[int, float]], bool]:
    """Return the ranking and whether chunk embeddings came from cache."""

    if any(chunk.index != position for position, chunk in enumerate(chunks)):
        raise ValueError("Chunk indices must match their list positions")

    query_embedding = model.encode(question, normalize_embeddings=True)
    chunk_embeddings, cache_hit = load_or_encode_chunks(
        chunks, model, model_name=model_name, revision=revision
    )
    # Normalized embeddings make this dot product equal cosine similarity.
    scores = chunk_embeddings @ query_embedding

    ranking = sorted(
        ((chunk.index, float(score)) for chunk, score in zip(chunks, scores)),
        key=lambda item: (-item[1], item[0]),
    )
    return ranking, cache_hit


def rank_chunks_bm25(
    question: str,
    chunks: list[Chunk],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[tuple[int, float]]:
    """Rank chunks with a small deterministic Okapi BM25 implementation."""

    documents = [TOKEN_PATTERN.findall(chunk.text.lower()) for chunk in chunks]
    query_terms = TOKEN_PATTERN.findall(question.lower())
    document_frequency = Counter(
        term for document in documents for term in set(document)
    )
    average_length = sum(map(len, documents)) / len(documents)
    scores = []

    for chunk, document in zip(chunks, documents):
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            frequency_in_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (len(documents) - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            length_normalization = k1 * (
                1 - b + b * len(document) / average_length
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1) / (frequency + length_normalization)
            )
        scores.append((chunk.index, score))

    return sorted(scores, key=lambda item: (-item[1], item[0]))


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Fuse rankings without making their incompatible scores comparable."""

    scores: Counter[int] = Counter()
    for ranking in rankings:
        for rank, (chunk_index, _score) in enumerate(ranking, start=1):
            scores[chunk_index] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
