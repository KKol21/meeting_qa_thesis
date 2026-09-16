"""Small, dependency-free bootstrap helpers."""

import random
from statistics import mean
from typing import Sequence


class Bootstrap:
    """Reusable bootstrap draws for equally sized observations."""

    def __init__(self, size: int, samples: int = 10_000, seed: int = 42):
        if size <= 0:
            raise ValueError("Cannot bootstrap an empty sample")
        if samples < 2:
            raise ValueError("At least two bootstrap samples are required")
        rng = random.Random(seed)
        self.size = size
        self.draws = [
            [rng.randrange(size) for _ in range(size)] for _ in range(samples)
        ]

    def mean(self, values: Sequence[float]) -> dict[str, object]:
        if len(values) != self.size:
            raise ValueError("Values do not match the bootstrap sample size")
        estimates = sorted(
            sum(values[index] for index in draw) / self.size
            for draw in self.draws
        )
        last = len(estimates) - 1
        return {
            "mean": mean(values),
            "ci95": [
                estimates[round(0.025 * last)],
                estimates[round(0.975 * last)],
            ],
        }

    def paired(
        self, left: Sequence[float], right: Sequence[float]
    ) -> dict[str, object]:
        if len(left) != len(right):
            raise ValueError("Paired samples must have equal length")
        return self.mean(
            [left_value - right_value for left_value, right_value in zip(left, right)]
        )


def bootstrap_mean(
    values: Sequence[float], samples: int = 10_000, seed: int = 42
) -> dict[str, object]:
    """Estimate a mean and percentile interval by resampling observations."""

    return Bootstrap(len(values), samples, seed).mean(values)


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    """Bootstrap the paired mean difference ``left - right``."""

    return Bootstrap(len(left), samples, seed).paired(left, right)
