"""Generate a compact Markdown report from completed ablation summaries."""

import argparse
import json
from pathlib import Path

from meeting_qa_chunking.config import load_run_config


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


def model_name(model: dict[str, object] | str) -> str:
    if isinstance(model, str):
        return model
    for field in ("name", "model", "tag"):
        if field in model:
            return str(model[field])
    raise ValueError("Model record has no name")


def evaluated_metrics(stage: dict[str, object], condition: str) -> tuple[dict, dict]:
    if "meeting_average" in stage:
        return stage["meeting_average"][condition], stage["question_average"][condition]
    metrics = stage["conditions"][condition]
    return metrics, metrics


def make_report(
    root: Path,
    answer_stage_names: list[str],
) -> str:
    retrieval = load_json(root / "retrieval" / "summary.json")
    evaluation = load_json(root / "evaluation" / "summary.json")
    answer_root = root / "answers"
    answer_summaries = {
        name: load_json(answer_root / name / "summary.json")
        for name in answer_stage_names
    }
    judge_model = model_name(evaluation["evaluation_config"]["judge"]["model"])
    candidate_models = {
        model_name(stage["answer_model"])
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
        rouge_averages = answer.get(
            "meeting_average_rouge_f1", answer.get("macro_average_rouge_f1")
        )
        rouge = rouge_averages["oracle"]
        metrics, distribution_metrics = evaluated_metrics(stage, "oracle")
        oracle_rows.append(
            {
                "name": stage["answer_model"]["tag"],
                "rouge1": rouge["rouge1"],
                "rouge2": rouge["rouge2"],
                "rougeL": rouge["rougeL"],
                "bert": metrics["bertscore"]["f1"],
                "judge": metrics["judge_mean"],
                "scores": distribution(distribution_metrics["judge_distribution"]),
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
    retrieval_averages = retrieval.get(
        "meeting_average", retrieval.get("macro_average")
    )
    rouge_averages = answer_stage.get(
        "meeting_average_rouge_f1",
        answer_stage.get("macro_average_rouge_f1"),
    )
    rows = []
    for name, config in retrieval["configurations"].items():
        retrieval_metrics = retrieval_averages[name]
        rouge = rouge_averages[name]
        answer_metrics, distribution_metrics = evaluated_metrics(
            retrieval_stage, name
        )
        rows.append(
            {
                "name": name,
                "chunker": config["chunker"],
                "retriever": config["retriever"],
                "words": config["evidence_words"],
                "precision": retrieval_metrics["precision"],
                "recall": retrieval_metrics["recall"],
                "mrr": retrieval_metrics.get(
                    "first_overlap_reciprocal_rank",
                    retrieval_metrics.get("reciprocal_rank"),
                ),
                "rouge1": rouge["rouge1"],
                "rouge2": rouge["rouge2"],
                "rougeL": rouge["rougeL"],
                "bert": answer_metrics["bertscore"]["f1"],
                "judge": answer_metrics["judge_mean"],
                "scores": distribution(distribution_metrics["judge_distribution"]),
            }
        )

    lines += [
        "",
        "## Retrieval and end-to-end comparison",
        "",
        (
            "Means are meeting-macro averages; judge 1/2/3 counts are question totals."
            if "meeting_average" in retrieval
            else "Means are question-macro averages."
        ) + " First-overlap MRR is retained only as a size-sensitive diagnostic.",
        "",
        "| Chunker | Retriever | Words | Precision | Recall | First-overlap MRR | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Judge | 1/2/3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['chunker']} | {row['retriever']} | {row['words']} | "
            f"{number(row['precision'])} | {number(row['recall'])} | {number(row['mrr'])} | "
            f"{number(row['rouge1'])} | {number(row['rouge2'])} | {number(row['rougeL'])} | "
            f"{number(row['bert'])} | {number(row['judge'])} | {row['scores']} |"
        )

    if retrieval.get("paired_meeting_average"):
        lines += [
            "",
            "## Paired meeting-level Lumber differences",
            "",
            "Positive values favour Lumber. Each value is the mean of within-meeting differences.",
            "",
            "| Comparison | Precision | Recall | ROUGE-L | BERTScore F1 | Judge |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for comparison, retrieval_delta in retrieval["paired_meeting_average"].items():
            rouge_delta = answer_stage["paired_meeting_average"][comparison]
            evaluation_delta = retrieval_stage["paired_meeting_average"][comparison]
            lines.append(
                f"| {comparison} | {number(retrieval_delta['precision'])} | "
                f"{number(retrieval_delta['recall'])} | "
                f"{number(rouge_delta['rougeL'])} | "
                f"{number(evaluation_delta['bertscore_f1'])} | "
                f"{number(evaluation_delta['judge_mean'])} |"
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
        "- Retrieval precision and recall are word-weighted against QMSum's annotated evidence spans. First-overlap MRR structurally favours larger chunks and is diagnostic only.",
        (
            "- Retrieval chooses evidence under the budget, then renders selected fragments chronologically for conversational coherence."
            if "meeting_average" in retrieval
            else "- This version-1 run presents evidence in retrieval-ranked order."
        ),
        "- ROUGE and BERTScore compare generated answers with the reference answers. The judge uses the reference answer and gold transcript evidence on a 1--3 scale.",
        "- The 1--3 judge compresses correctness, completeness, and grounding into one ordinal score; manual review remains necessary.",
        f"- {judge_note}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run = load_run_config(args.preset)
    if not run.run_evaluation:
        parser.error("the preset must enable evaluation to generate this report")
    root = run.output_root
    answer_stage_names = [stage.name for stage in run.answers]
    output = args.output or root / "report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        make_report(root, answer_stage_names),
        encoding="utf-8",
    )
    print(f"Ablation report: {output}")


if __name__ == "__main__":
    main()
