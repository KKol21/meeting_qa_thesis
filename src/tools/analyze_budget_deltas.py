"""Relate added retrieval coverage to answer changes at larger evidence budgets."""

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from meeting_qa_chunking.artifacts import read_json_object, write_json
from meeting_qa_chunking.config import load_run_config
from meeting_qa_chunking.uncertainty import Bootstrap


LOW_BUDGET = 1024
HIGH_BUDGET = 2048
METRICS = ("precision", "recall", "rougeL", "bertscore_f1", "judge")
DOWNSTREAM = ("rougeL", "bertscore_f1", "judge")
RECALL_BINS = (
    ("small", "< 0.10", lambda value: value < 0.10),
    ("moderate", "0.10 to < 0.25", lambda value: 0.10 <= value < 0.25),
    ("large", ">= 0.25", lambda value: value >= 0.25),
)
PRECISION_LOSS_BINS = (
    ("small", "< 0.05", lambda value: value < 0.05),
    ("moderate", "0.05 to < 0.15", lambda value: 0.05 <= value < 0.15),
    ("large", ">= 0.15", lambda value: value >= 0.15),
)


def average_ranks(values: list[float]) -> list[float]:
    """Return one-based ranks, assigning tied values their average rank."""

    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_scale = sum((x - left_mean) ** 2 for x in left)
    right_scale = sum((y - right_mean) ** 2 for y in right)
    if not left_scale or not right_scale:
        return None
    return numerator / math.sqrt(left_scale * right_scale)


def spearman(rows: list[dict], predictor: str, outcome: str) -> float | None:
    left = average_ranks([row[predictor] for row in rows])
    right = average_ranks([row["delta"][outcome] for row in rows])
    return correlation(left, right)


def interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    last = len(ordered) - 1
    return [ordered[round(0.025 * last)], ordered[round(0.975 * last)]]


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    output = defaultdict(list)
    for row in rows:
        output[row["meeting_id"]].append(row)
    return dict(output)


def summarize_rows(rows: list[dict], samples: int, seed: int) -> dict:
    """Macro-average questions within meetings, then bootstrap meetings."""

    by_meeting = grouped(rows)
    estimates = {}
    for metric in METRICS:
        values = [
            mean(row["delta"][metric] for row in meeting_rows)
            for meeting_rows in by_meeting.values()
        ]
        estimates[metric] = Bootstrap(len(values), samples, seed).mean(values)
    return {
        "question_count": len(rows),
        "meeting_count": len(by_meeting),
        "estimates": estimates,
    }


def summarize_bins(rows, definitions, value_key, samples, seed):
    output = []
    for name, label, accepts in definitions:
        selected = [row for row in rows if accepts(row[value_key])]
        output.append(
            {
                "name": name,
                "range": label,
                **summarize_rows(selected, samples, seed),
            }
        )
    return output


def cluster_spearman(rows, predictor, outcome, samples, seed):
    by_meeting = grouped(rows)
    meeting_rows = list(by_meeting.values())
    point = spearman(rows, predictor, outcome)
    estimates = []
    for draw in Bootstrap(len(meeting_rows), samples, seed).draws:
        sample = [row for index in draw for row in meeting_rows[index]]
        estimate = spearman(sample, predictor, outcome)
        if estimate is not None:
            estimates.append(estimate)
    if point is None or not estimates:
        raise ValueError(f"Undefined correlation for {predictor} and {outcome}")
    return {
        "spearman": point,
        "ci95": interval(estimates),
        "valid_bootstrap_samples": len(estimates),
    }


def evaluation_lookup(path: Path) -> dict[tuple[str, int, str], dict]:
    lookup = {}
    for record in read_json_object(path)["records"]:
        key = (
            record["meeting_id"],
            record["question_index"],
            record["condition"],
        )
        if key in lookup:
            raise ValueError(f"Duplicate evaluation record: {key}")
        lookup[key] = record
    return lookup


def load_rows(run, meeting_ids, answer_stage):
    evaluation = evaluation_lookup(
        run.evaluation_dir / f"{answer_stage.name}.json"
    )
    rows = []
    configurations = None
    for meeting_id in meeting_ids:
        retrieval = read_json_object(run.retrieval_dir / f"{meeting_id}.json")
        answers = read_json_object(
            run.answers_dir / answer_stage.name / f"{meeting_id}.json"
        )
        configurations = configurations or retrieval["configurations"]
        if retrieval["configurations"] != configurations:
            raise ValueError("Retrieval configurations differ between meetings")
        answer_questions = {
            item["question_index"]: item for item in answers["questions"]
        }
        for question in retrieval["questions"]:
            question_index = question["question_index"]
            answer = answer_questions.get(question_index)
            if answer is None:
                raise ValueError(f"Missing answer: {meeting_id} Q{question_index}")
            for chunker in run.retrieval.chunkers:
                for retriever in run.retrieval.retrievers:
                    low_name = f"{chunker}__{retriever}__w{LOW_BUDGET}"
                    high_name = f"{chunker}__{retriever}__w{HIGH_BUDGET}"
                    if low_name not in configurations or high_name not in configurations:
                        continue
                    low_retrieval = question["results"][low_name]
                    high_retrieval = question["results"][high_name]
                    low_answer = answer["results"][low_name]
                    high_answer = answer["results"][high_name]
                    low_eval = evaluation.get((meeting_id, question_index, low_name))
                    high_eval = evaluation.get((meeting_id, question_index, high_name))
                    if low_eval is None or high_eval is None:
                        raise ValueError(
                            f"Missing evaluation: {meeting_id} Q{question_index}"
                        )
                    low = {
                        "precision": low_retrieval["precision"],
                        "recall": low_retrieval["recall"],
                        "rougeL": low_answer["rouge_f1"]["rougeL"],
                        "bertscore_f1": low_eval["bertscore"]["f1"],
                        "judge": low_eval["judge"]["score"],
                    }
                    high = {
                        "precision": high_retrieval["precision"],
                        "recall": high_retrieval["recall"],
                        "rougeL": high_answer["rouge_f1"]["rougeL"],
                        "bertscore_f1": high_eval["bertscore"]["f1"],
                        "judge": high_eval["judge"]["score"],
                    }
                    rows.append(
                        {
                            "meeting_id": meeting_id,
                            "question_index": question_index,
                            "question": question["question"],
                            "chunker": chunker,
                            "retriever": retriever,
                            "low_condition": low_name,
                            "high_condition": high_name,
                            "low": low,
                            "high": high,
                            "delta": {
                                metric: high[metric] - low[metric]
                                for metric in METRICS
                            },
                            "precision_loss": low["precision"] - high["precision"],
                        }
                    )
    pairs_per_question = sum(
        1
        for spec in configurations.values()
        if spec["evidence_words"] == LOW_BUDGET
        and f"{spec['chunker']}__{spec['retriever']}__w{HIGH_BUDGET}"
        in configurations
    )
    question_count = sum(
        len(read_json_object(run.retrieval_dir / f"{mid}.json")["questions"])
        for mid in meeting_ids
    )
    expected = question_count * pairs_per_question
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} paired rows, found {len(rows)}")
    return rows, configurations, question_count


def format_estimate(estimate) -> str:
    low, high = estimate["ci95"]
    return f"{estimate['mean']:+.3f} [{low:+.3f}, {high:+.3f}]"


def make_report(result) -> str:
    lines = [
        "# Evidence-budget delta analysis",
        "",
        f"**Dataset:** QMSum {result['dataset_split']}; "
        f"{result['meeting_count']} meetings, {result['question_count']} questions.",
        "",
        f"**Comparison:** {HIGH_BUDGET:,} minus {LOW_BUDGET:,} evidence words. "
        "Questions are first averaged within each meeting, and meetings receive equal "
        f"weight. Brackets are percentile 95% meeting-cluster bootstrap intervals "
        f"({result['bootstrap_samples']:,} samples; seed {result['seed']}).",
        "",
        "## Budget effect in every retrieval environment",
        "",
        "| Chunker | Retriever | dPrecision | dRecall | dROUGE-L | dBERTScore F1 | dJudge |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["environments"].items():
        estimates = row["estimates"]
        lines.append(
            f"| {row['chunker']} | {row['retriever']} | "
            f"{format_estimate(estimates['precision'])} | "
            f"{format_estimate(estimates['recall'])} | "
            f"{format_estimate(estimates['rougeL'])} | "
            f"{format_estimate(estimates['bertscore_f1'])} | "
            f"{format_estimate(estimates['judge'])} |"
        )

    focus = result["focus"]
    lines += [
        "",
        f"## Focused proportionality check: {focus['chunker']} + {focus['retriever']}",
        "",
        "The bins were fixed before inspecting these summaries. They are descriptive, "
        "not optimized cut-points. Within each bin, questions are averaged within each "
        "contributing meeting before meetings are averaged.",
        "",
        "### By recall gain",
        "",
        "| Recall gain | Questions | Meetings | dRecall | dPrecision | dROUGE-L | dBERTScore F1 | dJudge |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in focus["recall_gain_bins"]:
        estimates = row["estimates"]
        lines.append(
            f"| {row['name']} ({row['range']}) | {row['question_count']} | "
            f"{row['meeting_count']} | {format_estimate(estimates['recall'])} | "
            f"{format_estimate(estimates['precision'])} | "
            f"{format_estimate(estimates['rougeL'])} | "
            f"{format_estimate(estimates['bertscore_f1'])} | "
            f"{format_estimate(estimates['judge'])} |"
        )
    lines += [
        "",
        "### By precision loss",
        "",
        "Precision loss is `precision at 1,024 - precision at 2,048`, so a larger "
        "positive value means more added noise under the larger budget. The small "
        "stratum also includes questions whose precision increased.",
        "",
        "| Precision loss | Questions | Meetings | dRecall | dPrecision | dROUGE-L | dBERTScore F1 | dJudge |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in focus["precision_loss_bins"]:
        estimates = row["estimates"]
        lines.append(
            f"| {row['name']} ({row['range']}) | {row['question_count']} | "
            f"{row['meeting_count']} | {format_estimate(estimates['recall'])} | "
            f"{format_estimate(estimates['precision'])} | "
            f"{format_estimate(estimates['rougeL'])} | "
            f"{format_estimate(estimates['bertscore_f1'])} | "
            f"{format_estimate(estimates['judge'])} |"
        )
    lines += [
        "",
        "### Question-level rank associations",
        "",
        "These are pooled-question Spearman correlations. Their intervals resample "
        "whole meetings, keeping all questions from a meeting together.",
        "",
        "| Predictor | Outcome | Spearman rho [95% CI] |",
        "|---|---|---:|",
    ]
    for name, row in focus["correlations"].items():
        low, high = row["ci95"]
        lines.append(
            f"| {row['predictor_label']} | {row['outcome_label']} | "
            f"{row['spearman']:+.3f} [{low:+.3f}, {high:+.3f}] |"
        )

    recall_large = next(row for row in focus["recall_gain_bins"] if row["name"] == "large")
    precision_large = next(
        row for row in focus["precision_loss_bins"] if row["name"] == "large"
    )
    environment = result["environments"][
        f"{focus['chunker']}__{focus['retriever']}"
    ]["estimates"]
    recall_judge = focus["correlations"]["recall_gain__judge"]
    loss_judge = focus["correlations"]["precision_loss__judge"]
    lines += [
        "",
        "## Reading the result",
        "",
        f"- In the focused setting, doubling the evidence budget increased recall by "
        f"{format_estimate(environment['recall'])}, while the judge changed by only "
        f"{format_estimate(environment['judge'])}. ROUGE-L changed by "
        f"{format_estimate(environment['rougeL'])} and BERTScore F1 by "
        f"{format_estimate(environment['bertscore_f1'])}.",
        f"- {recall_large['question_count']} questions had a recall gain of at least "
        f"0.25. Their mean judge change was "
        f"{format_estimate(recall_large['estimates']['judge'])}.",
        f"- {precision_large['question_count']} questions lost at least 0.15 precision. "
        f"Their mean judge change was "
        f"{format_estimate(precision_large['estimates']['judge'])}.",
        f"- Recall gain had a positive rank association with judge change "
        f"({recall_judge['spearman']:+.3f}, 95% CI "
        f"[{recall_judge['ci95'][0]:+.3f}, {recall_judge['ci95'][1]:+.3f}]); "
        f"precision loss had a negative association ({loss_judge['spearman']:+.3f}, "
        f"95% CI [{loss_judge['ci95'][0]:+.3f}, {loss_judge['ci95'][1]:+.3f}]). "
        "This is consistent with useful large coverage gains being offset when the "
        "additional context is mostly irrelevant.",
        "- A positive retrieval delta with a downstream interval spanning zero supports "
        "the interpretation that extra annotated evidence did not translate reliably "
        "into better answers under that condition; it does not prove the extra evidence "
        "was unused on every question.",
        "- Evidence precision and recall are word-level proxies based on QMSum's annotated "
        "turns. Judge scores are ordinal (1-3), although their paired differences are "
        "summarized numerically here. The associations are exploratory and not causal.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python src/tools/analyze_budget_deltas.py `",
        f"  --preset {result['preset']} `",
        f"  --output {result['output']} `",
        f"  --samples {result['bootstrap_samples']} --seed {result['seed']}",
        "```",
        "",
        "The sibling JSON contains every paired question-level row and all estimates "
        "used in this report.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focus-chunker", default="lumber")
    parser.add_argument("--focus-retriever", default="dense")
    args = parser.parse_args()

    run = load_run_config(args.preset)
    if LOW_BUDGET not in run.retrieval.evidence_budgets or HIGH_BUDGET not in run.retrieval.evidence_budgets:
        raise ValueError(f"Preset must contain budgets {LOW_BUDGET} and {HIGH_BUDGET}")
    answer_stage = next(stage for stage in run.answers if stage.source == "retrieval")
    meeting_ids = run.meeting_ids()
    rows, configurations, question_count = load_rows(run, meeting_ids, answer_stage)

    environments = {}
    for chunker in run.retrieval.chunkers:
        for retriever in run.retrieval.retrievers:
            selected = [
                row
                for row in rows
                if row["chunker"] == chunker and row["retriever"] == retriever
            ]
            if selected:
                environments[f"{chunker}__{retriever}"] = {
                    "chunker": chunker,
                    "retriever": retriever,
                    **summarize_rows(selected, args.samples, args.seed),
                }

    focus_rows = [
        row
        for row in rows
        if row["chunker"] == args.focus_chunker
        and row["retriever"] == args.focus_retriever
    ]
    if not focus_rows:
        raise ValueError("Focused chunker/retriever pair is absent")
    correlations = {}
    for predictor, predictor_label in (
        ("recall_gain", "Recall gain"),
        ("precision_loss", "Precision loss"),
    ):
        if predictor == "recall_gain":
            for row in focus_rows:
                row[predictor] = row["delta"]["recall"]
        for outcome in DOWNSTREAM:
            name = f"{predictor}__{outcome}"
            correlations[name] = {
                "predictor_label": predictor_label,
                "outcome_label": outcome,
                **cluster_spearman(
                    focus_rows, predictor, outcome, args.samples, args.seed
                ),
            }

    result = {
        "preset": str(args.preset).replace("\\", "/"),
        "output": str(args.output).replace("\\", "/"),
        "dataset_split": "validation" if run.data_dir.name == "val" else run.data_dir.name,
        "meeting_ids": meeting_ids,
        "meeting_count": len(meeting_ids),
        "question_count": question_count,
        "answer_stage": answer_stage.name,
        "budgets": {"low": LOW_BUDGET, "high": HIGH_BUDGET},
        "bootstrap_unit": "meeting",
        "bootstrap_samples": args.samples,
        "seed": args.seed,
        "configurations": configurations,
        "environments": environments,
        "focus": {
            "chunker": args.focus_chunker,
            "retriever": args.focus_retriever,
            "recall_gain_bins": summarize_bins(
                focus_rows,
                RECALL_BINS,
                "recall_gain",
                args.samples,
                args.seed,
            ),
            "precision_loss_bins": summarize_bins(
                focus_rows,
                PRECISION_LOSS_BINS,
                "precision_loss",
                args.samples,
                args.seed,
            ),
            "correlations": correlations,
        },
        "question_deltas": rows,
    }
    write_json(args.output, result)
    report = args.output.with_suffix(".md")
    report.write_text(make_report(result), encoding="utf-8")
    print(f"Budget-delta data: {args.output}")
    print(f"Budget-delta report: {report}")


if __name__ == "__main__":
    main()
