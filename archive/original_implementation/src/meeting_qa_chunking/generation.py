from __future__ import annotations

from .models import TextModel
from .retrieval import Evidence
from .schema import Query


ANSWER_INSTRUCTIONS = (
    "Answer the question using only the supplied meeting evidence. Give a direct, "
    "complete response that is only as long as the question requires. Preserve "
    "speaker attribution when it matters. If the evidence is insufficient, say so."
)


def build_answer_prompt(question: str, evidence: str) -> str:
    return (
        f"{ANSWER_INSTRUCTIONS}\n\n"
        f"Meeting evidence:\n{evidence.strip()}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Answer:"
    )


def generate_answer(
    model: TextModel,
    query: Query,
    evidence: Evidence,
    *,
    temperature: float = 0.0,
) -> str:
    answer = model.complete(
        build_answer_prompt(query.text, evidence.text),
        temperature=temperature,
    ).strip()
    if not answer:
        raise ValueError(f"Model returned an empty answer for query {query.id}")
    return answer
