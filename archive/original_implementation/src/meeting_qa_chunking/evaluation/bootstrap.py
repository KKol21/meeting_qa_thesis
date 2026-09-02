from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Sequence


@dataclass(frozen=True, slots=True)
class PairedScore:
    meeting_id: str
    candidate: float
    baseline: float


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    delta: float
    lower: float
    upper: float


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return sum(values) / len(values)


def _quantile(ordered: Sequence[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_meeting_bootstrap(
    scores: Sequence[PairedScore],
    *,
    samples: int = 2_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapInterval:
    if not scores or samples <= 0 or not 0 < confidence < 1:
        raise ValueError("Scores, samples, and confidence must define a valid bootstrap")

    grouped: dict[str, list[PairedScore]] = defaultdict(list)
    for score in scores:
        grouped[score.meeting_id].append(score)
    meeting_ids = tuple(grouped)
    rng = random.Random(seed)
    deltas: list[float] = []

    for _ in range(samples):
        sampled = rng.choices(meeting_ids, k=len(meeting_ids))
        differences = [
            score.candidate - score.baseline
            for meeting_id in sampled
            for score in grouped[meeting_id]
        ]
        deltas.append(_mean(differences))

    deltas.sort()
    tail = (1 - confidence) / 2
    observed = _mean([score.candidate - score.baseline for score in scores])
    return BootstrapInterval(
        observed,
        _quantile(deltas, tail),
        _quantile(deltas, 1 - tail),
    )
