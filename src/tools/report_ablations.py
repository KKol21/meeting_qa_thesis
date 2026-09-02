"""Generate a compact Markdown report from completed ablation summaries."""

import argparse
import html
import json
from pathlib import Path

from meeting_qa_chunking.evidence import reconstruct_evidence, render_evidence, render_gold_evidence
from meeting_qa_chunking.pipeline import build_chunk_sets
from meeting_qa_chunking.qmsum import load_meeting


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing ablation result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: float) -> str:
    return f"{value:.3f}"


def distribution(scores: dict[str, int]) -> str:
    return f"{scores['1']}/{scores['2']}/{scores['3']}"


def best(rows: list[dict[str, object]], metric: str) -> dict[str, object]:
    return max(rows, key=lambda row: row[metric])


def preformatted(text: str) -> str:
    return f"<pre>{html.escape(text)}</pre>"


def evaluation_lookup(root: Path, stage: str) -> dict[tuple[str, int, str], dict]:
    result = load_json(root / "evaluation" / f"{stage}.json")
    return {
        (record["meeting_id"], record["question_index"], record["condition"]): record
        for record in result["records"]
    }


def retrieved_evidence(
    meeting, retrieval: dict[str, object], lumber_dir: Path
) -> list[dict[str, str]]:
    chunk_sets = build_chunk_sets(
        meeting,
        lumber_dir / f"{meeting.id}.json",
        retrieval["fixed_chunk_words"],
    )
    prepared = []
    for question in retrieval["questions"]:
        evidence = {}
        for name, config in retrieval["configurations"].items():
            selected = reconstruct_evidence(
                question["results"][name],
                chunk_sets[config["chunker"]],
                config["evidence_words"],
            )
            evidence[name] = render_evidence(selected, meeting)
        prepared.append(evidence)
    return prepared


def make_review(
    root: Path, data_dir: Path, lumber_dir: Path, samples_per_condition: int
) -> str:
    sample_description = (
        "All questions" if samples_per_condition == 0
        else f"First {samples_per_condition} question(s)"
    )
    lines = [
        "# Ablation manual-review samples",
        "",
        f"{sample_description} per ablation condition, in dataset order.",
        "",
        "Judge scores: 1 = invalid/incorrect, 2 = partially correct, 3 = correct.",
        "",
    ]
    for summary_path in sorted((root / "answers").glob("*/summary.json")):
        stage = summary_path.parent.name
        summary = load_json(summary_path)
        evaluations = evaluation_lookup(root, stage)
        counts = {condition: 0 for condition in summary["conditions"]}
        lines += [f"## {stage}", ""]

        for meeting_id in summary["meeting_ids"]:
            if samples_per_condition and all(
                count >= samples_per_condition for count in counts.values()
            ):
                break
            meeting = load_meeting(data_dir / f"{meeting_id}.json")
            answers = load_json(summary_path.parent / f"{meeting_id}.json")
            if summary["source"] == "retrieval":
                retrieval = load_json(root / "retrieval" / f"{meeting_id}.json")
                evidence_by_question = retrieved_evidence(
                    meeting, retrieval, lumber_dir
                )
            else:
                evidence_by_question = [
                    {"oracle": render_gold_evidence(question, meeting)[0]}
                    for question in meeting.questions
                ]

            for question_result in answers["questions"]:
                question_index = question_result["question_index"]
                question = meeting.questions[question_index]
                gold_evidence = render_gold_evidence(question, meeting)[0]
                for condition, result in question_result["results"].items():
                    if (
                        samples_per_condition
                        and counts[condition] >= samples_per_condition
                    ):
                        continue
                    counts[condition] += 1
                    evaluation = evaluations[(meeting_id, question_index, condition)]
                    judge = evaluation["judge"]
                    bert = evaluation["bertscore"]["f1"]
                    rouge = result["rouge_f1"]
                    lines += [
                        "<details>",
                        f"<summary><strong>{condition}</strong> · {meeting_id} Q{question_index}: {html.escape(question.text)}</summary>",
                        "",
                        "**Reference answer**",
                        preformatted(question.reference_answer),
                        "**Gold/oracle transcript span**",
                        preformatted(gold_evidence),
                    ]
                    if summary["source"] == "retrieval":
                        lines += [
                            "**Retrieved transcript span**",
                            preformatted(evidence_by_question[question_index][condition]),
                        ]
                    lines += [
                        "**Generated answer**",
                        preformatted(result["answer"]),
                        (
                            f"**Scores:** judge {judge['score']}/3 · BERTScore F1 {number(bert)} · "
                            f"ROUGE-1/2/L {number(rouge['rouge1'])}/{number(rouge['rouge2'])}/{number(rouge['rougeL'])}"
                        ),
                        "",
                        f"**Judge reason:** {html.escape(judge['reason'])}",
                        "",
                        "</details>",
                        "",
                    ]
    return "\n".join(lines)


def make_report(root: Path, review_name: str = "review.md") -> str:
    retrieval = load_json(root / "retrieval" / "summary.json")
    evaluation = load_json(root / "evaluation" / "summary.json")
    answer_root = root / "answers"
    answer_summaries = {
        path.parent.name: load_json(path)
        for path in sorted(answer_root.glob("*/summary.json"))
    }
    judge_model = evaluation["evaluation_config"]["judge"]["model"]
    candidate_models = {
        stage["answer_model"]["model"]
        for stage in evaluation["stages"].values()
    }
    if judge_model in candidate_models:
        judge_note = (
            f"The judge checkpoint `{judge_model}` is also used as a candidate "
            "model; report this as a possible source of bias."
        )
    else:
        judge_note = (
            f"The judge checkpoint is `{judge_model}`, separate from the "
            "candidate checkpoints."
        )

    meeting_count = retrieval["meeting_count"]
    question_count = retrieval["question_count"]
    lines = [
        "# Ablation report",
        "",
        f"**Scope:** {meeting_count} meeting(s), {question_count} question(s).",
        "",
        f"Detailed examples for manual inspection: [{review_name}]({review_name})",
        "",
    ]
    if meeting_count == 1:
        lines += [
            "> This is a smoke test. Treat rankings as pipeline validation, not experimental conclusions.",
            "",
        ]

    lines += [
        "## Oracle answer-model comparison",
        "",
        "Gold evidence is supplied here, isolating answer-model performance from retrieval.",
        "",
        "| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Judge mean | Judge 1/2/3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    oracle_rows = []
    for stage_name, stage in evaluation["stages"].items():
        if stage["source"] != "oracle":
            continue
        answer = answer_summaries[stage_name]
        rouge = answer["macro_average_rouge_f1"]["oracle"]
        metrics = stage["conditions"]["oracle"]
        oracle_rows.append(
            {
                "name": stage["answer_model"]["tag"],
                "rouge1": rouge["rouge1"],
                "rouge2": rouge["rouge2"],
                "rougeL": rouge["rougeL"],
                "bert": metrics["bertscore"]["f1"],
                "judge": metrics["judge_mean"],
                "scores": distribution(metrics["judge_distribution"]),
            }
        )
    for row in oracle_rows:
        lines.append(
            f"| {row['name']} | {number(row['rouge1'])} | {number(row['rouge2'])} | "
            f"{number(row['rougeL'])} | {number(row['bert'])} | "
            f"{number(row['judge'])} | {row['scores']} |"
        )

    retrieval_stage = next(
        stage for stage in evaluation["stages"].values()
        if stage["source"] == "retrieval"
    )
    answer_stage = next(
        stage for stage in answer_summaries.values()
        if stage["source"] == "retrieval"
    )
    rows = []
    for name, config in retrieval["configurations"].items():
        retrieval_metrics = retrieval["macro_average"][name]
        rouge = answer_stage["macro_average_rouge_f1"][name]
        answer_metrics = retrieval_stage["conditions"][name]
        rows.append(
            {
                "name": name,
                "chunker": config["chunker"],
                "retriever": config["retriever"],
                "words": config["evidence_words"],
                "precision": retrieval_metrics["precision"],
                "recall": retrieval_metrics["recall"],
                "mrr": retrieval_metrics["reciprocal_rank"],
                "rouge1": rouge["rouge1"],
                "rouge2": rouge["rouge2"],
                "rougeL": rouge["rougeL"],
                "bert": answer_metrics["bertscore"]["f1"],
                "judge": answer_metrics["judge_mean"],
                "scores": distribution(answer_metrics["judge_distribution"]),
            }
        )

    lines += [
        "",
        "## Retrieval and end-to-end comparison",
        "",
        "| Chunker | Retriever | Words | Precision | Recall | MRR | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Judge | 1/2/3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['chunker']} | {row['retriever']} | {row['words']} | "
            f"{number(row['precision'])} | {number(row['recall'])} | {number(row['mrr'])} | "
            f"{number(row['rouge1'])} | {number(row['rouge2'])} | {number(row['rougeL'])} | "
            f"{number(row['bert'])} | {number(row['judge'])} | {row['scores']} |"
        )

    lines += ["", "## Best observed configurations", ""]
    for label, metric in (
        ("Retrieval recall", "recall"),
        ("ROUGE-L", "rougeL"),
        ("BERTScore F1", "bert"),
        ("LLM judge", "judge"),
    ):
        winner = best(rows, metric)
        lines.append(
            f"- **{label}:** `{winner['name']}` ({number(winner[metric])})"
        )

    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Retrieval precision and recall are word-weighted against QMSum's annotated evidence spans; MRR measures the first relevant chunk rank.",
        "- ROUGE and BERTScore compare generated answers with the reference answers. The judge uses the reference answer and gold transcript evidence on a 1–3 scale.",
        f"- {judge_note}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--samples-per-condition", type=int, default=10)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/raw/qmsum/data/ALL/val")
    )
    parser.add_argument("--lumber-dir", type=Path, default=Path("runs/lumber/qmsum"))
    args = parser.parse_args()
    if args.samples_per_condition < 0:
        parser.error("--samples-per-condition must be zero or greater")

    output = args.output or args.root / "report.md"
    review_output = args.review_output or args.root / "review.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    review_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(make_report(args.root, review_output.name), encoding="utf-8")
    review_output.write_text(
        make_review(
            args.root,
            args.data_dir,
            args.lumber_dir,
            args.samples_per_condition,
        ),
        encoding="utf-8",
    )
    print(f"Ablation report: {output}")
    print(f"Manual review: {review_output}")


if __name__ == "__main__":
    main()
