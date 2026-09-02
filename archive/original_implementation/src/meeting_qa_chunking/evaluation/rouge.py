from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RougeScores:
    rouge1: float
    rouge2: float
    rougeL: float


def score_rouge(reference: str, prediction: str) -> RougeScores:
    try:
        from rouge_score.rouge_scorer import RougeScorer
    except ImportError as error:
        raise RuntimeError(
            "Install the experiment dependencies: pip install -e '.[experiment]'"
        ) from error

    scores = RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    ).score(reference, prediction)
    return RougeScores(*(scores[name].fmeasure for name in ("rouge1", "rouge2", "rougeL")))
