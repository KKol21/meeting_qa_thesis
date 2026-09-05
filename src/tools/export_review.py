"""Export selected meetings from an ablation run as JSON and Markdown."""

import argparse
import html
import json
from pathlib import Path

from meeting_qa_chunking.artifacts import write_json
from meeting_qa_chunking.evidence import render_gold_evidence
from meeting_qa_chunking.evidence_preparation import prepare_retrieved_evidence
from meeting_qa_chunking.qmsum import load_meeting


DEFAULT_DATA_DIR = Path("data/raw/qmsum/data/ALL/val")
DEFAULT_RUNS_DIR = Path("runs/ablations")
STAGES = {"oracle-14b": "oracle", "retrieval-14b": "retrieval"}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing run artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def index_by(items: list[dict], *fields: str) -> dict[tuple, dict]:
    indexed = {tuple(item[field] for field in fields): item for item in items}
    if len(indexed) != len(items):
        raise ValueError(f"Duplicate records for fields: {', '.join(fields)}")
    return indexed


def load_stages(
    root: Path, meeting_ids: list[str], retrieval_condition: str
) -> list[dict]:
    answer_root = root / "answers"
    stages = []
    for name, source in STAGES.items():
        answer_dir = answer_root / name
        summary = load_json(answer_dir / "summary.json")
        if summary["source"] != source:
            raise ValueError(f"Unexpected source for answer stage {name}")
        missing = set(meeting_ids) - set(summary["meeting_ids"])
        if missing:
            raise ValueError(
                f"Answer stage {name} lacks meetings: {sorted(missing)}"
            )
        selected = "oracle" if source == "oracle" else retrieval_condition
        conditions = summary["conditions"]
        if selected not in conditions:
            raise ValueError(f"Unknown {source} condition: {selected}")
        evaluation = load_json(root / "evaluation" / f"{name}.json")
        stages.append(
            {
                "name": name,
                "directory": answer_dir,
                "summary": summary,
                "conditions": {selected: conditions[selected]},
                "evaluations": index_by(
                    evaluation["records"],
                    "meeting_id",
                    "question_index",
                    "condition",
                ),
            }
        )
    return stages


def build_review(
    run_id: str,
    root: Path,
    data_dir: Path,
    meeting_ids: list[str],
    retrieval_condition: str,
) -> dict:
    if len(set(meeting_ids)) != len(meeting_ids):
        raise ValueError("Meeting IDs must be unique")

    stages = load_stages(root, meeting_ids, retrieval_condition)
    stage_metadata = {
        stage["name"]: {
            "source": stage["summary"]["source"],
            "answer_model": stage["summary"]["answer_model"],
            "conditions": stage["conditions"],
        }
        for stage in stages
    }
    meetings = []

    for meeting_id in meeting_ids:
        meeting = load_meeting(data_dir / f"{meeting_id}.json")
        stage_answers = {
            stage["name"]: index_by(
                load_json(stage["directory"] / f"{meeting_id}.json")["questions"],
                "question_index",
            )
            for stage in stages
        }

        retrieval_path = root / "retrieval" / f"{meeting_id}.json"
        retrieval = load_json(retrieval_path)
        retrieval_questions = index_by(retrieval["questions"], "question_index")
        _conditions, rendered_retrieval = prepare_retrieved_evidence(
            meeting,
            retrieval_path,
            root / "segmentation",
            [retrieval_condition],
        )

        questions = []
        for question_index, question in enumerate(meeting.questions):
            gold_text, gold_turn_ids = render_gold_evidence(question, meeting)
            results = []
            for stage in stages:
                answer = stage_answers[stage["name"]][(question_index,)]
                if answer["question"] != question.text:
                    raise ValueError(
                        f"Question mismatch in {stage['name']} Q{question_index}"
                    )
                for condition in stage["conditions"]:
                    generated = answer["results"][condition]
                    key = (meeting_id, question_index, condition)
                    evaluation = stage["evaluations"].get(key)
                    if evaluation is None:
                        raise ValueError(f"Missing evaluation record: {key}")

                    retrieval_scores = None
                    retrieved = None
                    if stage["summary"]["source"] == "retrieval":
                        saved = retrieval_questions[(question_index,)]["results"][
                            condition
                        ]
                        retrieval_scores = {
                            name: value
                            for name, value in saved.items()
                            if name != "selected_chunk_indices"
                        }
                        retrieved = {
                            "text": rendered_retrieval[question_index][condition][
                                "text"
                            ],
                            "evidence_order": retrieval["evidence_order"],
                            "selected_chunk_indices": saved["selected_chunk_indices"],
                        }

                    results.append(
                        {
                            "answer_stage": stage["name"],
                            "condition": condition,
                            "retrieved": retrieved,
                            "model_answer": generated["answer"],
                            "scores": {
                                "retrieval": retrieval_scores,
                                "rouge_f1": generated["rouge_f1"],
                                "bertscore": evaluation["bertscore"],
                                "judge": evaluation["judge"]["score"],
                            },
                            "judge_reasoning": evaluation["judge"]["reason"],
                        }
                    )

            questions.append(
                {
                    "question_index": question_index,
                    "question": question.text,
                    "reference_answer": question.reference_answer,
                    "gold": {
                        "turn_ranges": question.relevant_turn_ranges,
                        "turn_ids": gold_turn_ids,
                        "text": gold_text,
                    },
                    "results": results,
                }
            )
        meetings.append({"meeting_id": meeting_id, "questions": questions})

    return {
        "run_id": run_id,
        "retrieval_condition": retrieval_condition,
        "meeting_ids": meeting_ids,
        "answer_stages": stage_metadata,
        "meetings": meetings,
    }


def block(text: str) -> str:
    cleaned = "\n".join(line.rstrip() for line in text.splitlines())
    return f"<pre>{html.escape(cleaned)}</pre>"


def number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def score_line(scores: dict) -> str:
    retrieval = scores["retrieval"]
    parts = []
    if retrieval:
        parts.append(
            "retrieval P/R/MRR "
            f"{number(retrieval['precision'])}/{number(retrieval['recall'])}/"
            f"{number(retrieval['first_overlap_reciprocal_rank'])}"
        )
    rouge = scores["rouge_f1"]
    bert = scores["bertscore"]
    parts += [
        f"ROUGE-1/2/L {number(rouge['rouge1'])}/{number(rouge['rouge2'])}/{number(rouge['rougeL'])}",
        f"BERTScore P/R/F1 {number(bert['precision'])}/{number(bert['recall'])}/{number(bert['f1'])}",
        f"judge {scores['judge']}/3",
    ]
    return " | ".join(parts)


def make_markdown(review: dict) -> str:
    lines = [
        f"# Selected review: {review['run_id']}",
        "",
        f"Meetings: {', '.join(review['meeting_ids'])}",
        "",
        "Judge scores: 1 = invalid/incorrect, 2 = partially correct, 3 = correct.",
        "",
    ]
    for meeting in review["meetings"]:
        lines += [f"## {meeting['meeting_id']}", ""]
        for question in meeting["questions"]:
            lines += [
                "<details>",
                (
                    f"<summary><strong>Q{question['question_index']}</strong>: "
                    f"{html.escape(question['question'])}</summary>"
                ),
                "",
                "**Reference answer**",
                block(question["reference_answer"]),
                "**Gold transcript evidence**",
                block(question["gold"]["text"]),
            ]
            for result in question["results"]:
                lines += [
                    f"### {result['answer_stage']} / {result['condition']}",
                    "",
                ]
                if result["retrieved"]:
                    lines += [
                        "**Retrieved transcript evidence**",
                        block(result["retrieved"]["text"]),
                    ]
                lines += [
                    "**Model answer**",
                    block(result["model_answer"]),
                    f"**Scores:** {score_line(result['scores'])}",
                    "",
                    f"**Judge reasoning:** {html.escape(result['judge_reasoning'])}",
                    "",
                ]
            lines += ["</details>", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run identifier, e.g. full")
    parser.add_argument(
        "--condition",
        "--specification",
        dest="condition",
        required=True,
        help="Retrieval condition, e.g. lumber__dense__w512",
    )
    parser.add_argument("--meetings", nargs="*")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()

    root = args.runs_dir / args.run
    meeting_ids = args.meetings or load_json(
        root / "answers" / "retrieval-14b" / "summary.json"
    )["meeting_ids"]
    review = build_review(
        args.run, root, args.data_dir, meeting_ids, args.condition
    )
    output_prefix = args.output_prefix or root / "review-selection"
    json_output = output_prefix.with_suffix(".json")
    markdown_output = output_prefix.with_suffix(".md")
    write_json(json_output, review)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(make_markdown(review), encoding="utf-8")
    print(f"JSON review: {json_output}")
    print(f"Markdown review: {markdown_output}")


if __name__ == "__main__":
    main()
