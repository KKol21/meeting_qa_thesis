"""Compare the gold-only score with independent faithfulness and sufficiency checks."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from meeting_qa_chunking.artifacts import read_answers, write_json
from meeting_qa_chunking.config import load_run_config
from meeting_qa_chunking.evidence import render_gold_evidence
from meeting_qa_chunking.evidence_preparation import prepare_retrieved_evidence
from meeting_qa_chunking.judging import (
    FAITHFULNESS_INSTRUCTION,
    SUFFICIENCY_INSTRUCTION,
    build_axis_retry_prompt,
    build_faithfulness_prompt,
    build_sufficiency_prompt,
    parse_axis_judgment,
)
from meeting_qa_chunking.qmsum import load_meeting


def find_original_judgment(path: Path, meeting: str, question: int, condition: str):
    saved = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        record["judge"]
        for record in saved["records"]
        if record["meeting_id"] == meeting
        and record["question_index"] == question
        and record["condition"] == condition
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one original judgment, found {len(matches)}")
    return matches[0]


def call_axis(judge, prompt: str, issue_field: str) -> dict[str, object]:
    response = judge(prompt)
    try:
        result = parse_axis_judgment(response, issue_field)
    except ValueError:
        judge.discard_last_response()
        response = judge(build_axis_retry_prompt(prompt, response))
        try:
            result = parse_axis_judgment(response, issue_field)
        except ValueError:
            judge.discard_last_response()
            raise
    return {**result, "raw_response": response, "cache_hit": judge.last_cache_hit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--meeting", required=True)
    parser.add_argument("--question-index", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--answer-stage", default="retrieval-14b")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run = load_run_config(args.preset)
    meeting = load_meeting(run.data_dir / f"{args.meeting}.json")
    if not 0 <= args.question_index < len(meeting.questions):
        parser.error("question index is outside this meeting")
    question = meeting.questions[args.question_index]

    answers = read_answers(
        run.answers_dir / args.answer_stage / f"{args.meeting}.json"
    )
    question_result = answers["questions"][args.question_index]
    if question_result["question_index"] != args.question_index:
        raise ValueError("Answer question indices are not contiguous")
    try:
        candidate = question_result["results"][args.condition]["answer"]
    except KeyError:
        parser.error(f"condition not found in answer artifact: {args.condition}")

    _conditions, prepared = prepare_retrieved_evidence(
        meeting,
        run.retrieval_dir / f"{args.meeting}.json",
        run.lumber_dir,
        [args.condition],
    )
    retrieved = prepared[args.question_index][args.condition]
    gold_text, gold_turn_ids = render_gold_evidence(question, meeting)
    original = find_original_judgment(
        run.evaluation_dir / f"{args.answer_stage}.json",
        args.meeting,
        args.question_index,
        args.condition,
    )

    spec = run.evaluation
    from meeting_qa_chunking.local_model import LocalChatModel

    judge = LocalChatModel(
        model_name=spec.judge_model.name,
        revision=spec.judge_model.revision,
        max_new_tokens=spec.judge_max_new_tokens,
        seed=spec.judge_seed,
        temperature=spec.judge_temperature,
        cache_dir=Path(".cache/grounded-judgments"),
        prequantized=spec.judge_model.prequantized,
    )
    faithfulness = call_axis(
        judge,
        build_faithfulness_prompt(question.text, retrieved["text"], candidate),
        "unsupported_claims",
    )
    sufficiency = call_axis(
        judge,
        build_sufficiency_prompt(
            question.text, question.reference_answer, retrieved["text"]
        ),
        "unavailable_reference_points",
    )

    result = {
        "preset": str(args.preset),
        "meeting_id": args.meeting,
        "question_index": args.question_index,
        "condition": args.condition,
        "question": question.text,
        "reference_answer": question.reference_answer,
        "gold_evidence": {"text": gold_text, "turn_ids": gold_turn_ids},
        "retrieved_evidence": retrieved,
        "candidate_answer": candidate,
        "original_gold_only_judge": original,
        "evidence_aware_judge": {
            "faithfulness": faithfulness,
            "evidence_sufficiency": sufficiency,
        },
        "judge_config": {
            "model": asdict(spec.judge_model),
            "max_new_tokens": spec.judge_max_new_tokens,
            "temperature": spec.judge_temperature,
            "seed": spec.judge_seed,
            "faithfulness_prompt": FAITHFULNESS_INSTRUCTION,
            "sufficiency_prompt": SUFFICIENCY_INSTRUCTION,
        },
    }
    output = args.output or (
        run.output_root
        / "grounded-judge"
        / f"{args.meeting}-q{args.question_index}-{args.condition}.json"
    )
    write_json(output, result)
    print(f"Original judge: {original['score']} - {original['reason']}")
    print(
        "Evidence-aware judge: "
        f"faithfulness={faithfulness['score']}, "
        f"evidence_sufficiency={sufficiency['score']}"
    )
    print(f"Faithfulness: {faithfulness['reason']}")
    print(f"Sufficiency: {sufficiency['reason']}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
