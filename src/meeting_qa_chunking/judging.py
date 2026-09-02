"""Dataset-neutral prompt and parser for three-level answer judging."""

import json
import re

from .prompt_files import load_prompt


JUDGE_INSTRUCTION = load_prompt("judge.txt")


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
