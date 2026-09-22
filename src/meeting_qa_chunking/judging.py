"""Dataset-neutral prompt and parser for three-level answer judging."""

import json
import re

from .prompt_files import load_prompt


JUDGE_INSTRUCTION = load_prompt("judge.txt")
FAITHFULNESS_INSTRUCTION = load_prompt("judge_faithfulness.txt")
SUFFICIENCY_INSTRUCTION = load_prompt("judge_sufficiency.txt")


def build_judge_prompt(
    question: str,
    reference_answer: str,
    gold_evidence: str,
    candidate_answer: str,
) -> str:
    return (
        f"{JUDGE_INSTRUCTION}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Gold transcript evidence:\n{gold_evidence}\n\n"
        f"Candidate answer:\n{candidate_answer}"
    )


def build_judge_retry_prompt(prompt: str, invalid_response: str) -> str:
    """Constrain a second attempt after malformed judge output."""

    return (
        f"{prompt}\n\nYour previous response was invalid: {invalid_response!r}\n"
        'Return only JSON in this form: {"score": 1, "reason": "..."}'
    )


def build_axis_retry_prompt(prompt: str, invalid_response: str) -> str:
    return (
        f"{prompt}\n\nYour previous response was invalid: {invalid_response!r}\n"
        "Return only the exact JSON object requested above."
    )


def parse_judgment(response: str) -> tuple[int, str]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response):
        try:
            value, _end = decoder.raw_decode(response[match.start() :])
        except json.JSONDecodeError:
            continue
        score = value.get("score") if isinstance(value, dict) else None
        reason = value.get("reason") if isinstance(value, dict) else None
        if isinstance(score, int) and not isinstance(score, bool) and score in (1, 2, 3):
            return score, str(reason or "No reason supplied").strip()

    score_match = re.search(r"[\"']?score[\"']?\s*[:=]\s*([123])", response, re.I)
    if score_match:
        return int(score_match.group(1)), response.strip()
    raise ValueError(f"Could not parse judge response: {response!r}")


def build_faithfulness_prompt(
    question: str, supplied_evidence: str, candidate_answer: str
) -> str:
    return (
        f"{FAITHFULNESS_INSTRUCTION}\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence supplied to the answering model:\n{supplied_evidence}\n\n"
        f"Candidate answer:\n{candidate_answer}"
    )


def build_sufficiency_prompt(
    question: str, reference_answer: str, supplied_evidence: str
) -> str:
    return (
        f"{SUFFICIENCY_INSTRUCTION}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Evidence supplied to the answering model:\n{supplied_evidence}"
    )


def parse_axis_judgment(response: str, issue_field: str) -> dict[str, object]:
    """Parse one independent evidence-quality judgment."""

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response):
        try:
            value, _end = decoder.raw_decode(response[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        score = value.get("score")
        issues = value.get(issue_field)
        reason = value.get("reason")
        if (
            isinstance(score, int)
            and not isinstance(score, bool)
            and score in (1, 2, 3)
            and isinstance(issues, list)
            and all(isinstance(issue, str) and issue.strip() for issue in issues)
            and (score == 3) == (not issues)
            and isinstance(reason, str)
            and reason.strip()
        ):
            return {
                "score": score,
                issue_field: issues,
                "reason": reason.strip(),
            }
    raise ValueError(f"Could not parse {issue_field} judgment: {response!r}")
