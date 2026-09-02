from __future__ import annotations

from dataclasses import dataclass
import re

from ..models import TextModel
from ..schema import Query


# English QA rubric from utter-project/ELITR-Bench, utils/common.py (CC BY 4.0).
ELITR_JUDGE_TASK = (
    "You are provided below with a question, a response to evaluate, a reference "
    "answer that gets the maximum score of 10, and a score rubric representing "
    "evaluation criteria.\n"
    "1. Write a detailed feedback that assess the quality of the response strictly "
    "based on the given score rubric, not evaluating in general.\n"
    "2. After writing a feedback, write a score that is an integer between 1 and "
    "10. You should refer to the score rubric.\n"
    "3. The output format should first include the feedback and then indicate the "
    "integer score in \\boxed{}.\n"
    "4. Please do not generate any other opening, closing, and explanations."
)
ELITR_JUDGE_RUBRIC = (
    "[Does the response to evaluate correctly address the given question based on "
    "the elements provided by the reference answer? The response should include "
    "the elements of the reference answer and should also avoid adding unnecessary "
    "elements or being too verbose.]\n"
    "Score 1: The response to evaluate is incorrect and misses all the elements of "
    "the reference answer.\n"
    "Score 2: The response to evaluate indicates insufficient knowledge to answer "
    "the question even though the reference answer states otherwise.\n"
    "Score 3-4: The response to evaluate contains some elements vaguely related to "
    "the reference answer.\n"
    "Score 5-6: The response to evaluate is partially correct and/or covers only a "
    "part of the reference answer.\n"
    "Score 7-8: The response to evaluate contains most of the reference answer but "
    "delivers it in an indirect and/or overly verbose way.\n"
    "Score 9: The response to evaluate includes the reference answer but it is more "
    "verbose and adds unnecessary elements.\n"
    "Score 10: The response to evaluate is essentially equivalent to the reference "
    "answer."
)
_BOXED_SCORE = re.compile(r"\\boxed\{(\d+)\}")


@dataclass(frozen=True, slots=True)
class ELITRJudgment:
    score: int
    feedback: str
    raw_response: str


def build_elitr_judge_prompt(question: str, response: str, reference: str) -> str:
    return (
        f"### Task description:\n{ELITR_JUDGE_TASK}\n\n"
        f"### Question:\n{question.strip()}\n\n"
        f"### Response to evaluate:\n{response.strip()}\n\n"
        f"### Reference answer (score 10):\n{reference.strip()}\n\n"
        f"### Score rubric:\n{ELITR_JUDGE_RUBRIC}\n\n"
        "### Feedback:\n"
    )


def parse_elitr_judgment(response: str) -> ELITRJudgment:
    match = _BOXED_SCORE.search(response)
    if not match:
        raise ValueError("ELITR judge response has no boxed score")
    score = int(match.group(1))
    if not 1 <= score <= 10:
        raise ValueError(f"ELITR judge score {score} is outside [1, 10]")
    return ELITRJudgment(score, response[: match.start()].strip(), response)


def judge_elitr_answer(
    model: TextModel,
    query: Query,
    answer: str,
    *,
    temperature: float = 0.0,
    max_attempts: int = 2,
) -> ELITRJudgment:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    prompt = build_elitr_judge_prompt(query.text, answer, query.reference_answer)
    last_error: ValueError | None = None
    for _ in range(max_attempts):
        response = model.complete(prompt, temperature=temperature)
        try:
            return parse_elitr_judgment(response)
        except ValueError as error:
            last_error = error
    raise ValueError(f"No valid ELITR judgment after {max_attempts} attempts") from last_error
