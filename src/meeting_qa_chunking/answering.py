"""Shared prompt for answering from retrieved meeting evidence."""

from rouge_score import rouge_scorer

from .prompt_files import load_prompt


ANSWER_INSTRUCTION = load_prompt("answer.txt")
ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")


def build_answer_prompt(question: str, evidence: str) -> str:
    return (
        f"{ANSWER_INSTRUCTION}\n\n"
        f"Question:\n{question}\n\n"
        f"Meeting excerpts:\n{evidence}"
    )


def score_answer(
    scorer: rouge_scorer.RougeScorer,
    reference: str,
    answer: str,
) -> dict[str, float]:
    scores = scorer.score(reference, answer)
    return {name: scores[name].fmeasure for name in ROUGE_TYPES}
