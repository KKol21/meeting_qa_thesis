"""Meeting-level bootstrap uncertainty for a completed ablation run."""

import argparse
from itertools import combinations
from pathlib import Path

from meeting_qa_chunking.artifacts import read_json_object, write_json
from meeting_qa_chunking.config import BASELINE_CHUNKERS, load_run_config
from meeting_qa_chunking.uncertainty import Bootstrap


RETRIEVAL_METRICS = {
    "precision": "precision",
    "recall": "recall",
    "mrr": "first_overlap_reciprocal_rank",
}
ANSWER_METRICS = {
    "rouge1": "rouge1",
    "rouge2": "rouge2",
    "rougeL": "rougeL",
}


def require_meetings(summary, meeting_ids, label):
    if summary.get("meeting_ids") != meeting_ids:
        raise ValueError(f"{label} uses a different meeting list or order")


def stage_values(answer, evaluation, meeting_ids, condition):
    """Collect downstream metrics in stable meeting order."""

    values = {name: [] for name in (*ANSWER_METRICS, "bertscore_f1", "judge")}
    for meeting_id in meeting_ids:
        answer_row = answer["per_meeting"][meeting_id][condition]
        evaluation_row = evaluation["per_meeting"][meeting_id][condition]
        for name, field in ANSWER_METRICS.items():
            values[name].append(answer_row[field])
        values["bertscore_f1"].append(evaluation_row["bertscore"]["f1"])
        values["judge"].append(evaluation_row["judge_mean"])
    return values


def estimate_metrics(values, bootstrap):
    return {
        metric: bootstrap.mean(metric_values)
        for metric, metric_values in values.items()
    }


def contrast_metrics(left, right, bootstrap):
    shared = left.keys() & right.keys()
    return {
        metric: bootstrap.paired(left[metric], right[metric])
        for metric in shared
    }


def average_lumber_effects(condition_values, configurations, bootstrap):
    """Average paired effects over retriever x budget environments per meeting."""

    lumber = [
        name for name, spec in configurations.items() if spec["chunker"] == "lumber"
    ]
    output = {}
    for baseline in BASELINE_CHUNKERS:
        meeting_effects = []
        for meeting_index in range(bootstrap.size):
            differences = []
            for lumber_name in lumber:
                baseline_name = lumber_name.replace("lumber__", f"{baseline}__", 1)
                if baseline_name not in condition_values:
                    raise ValueError(f"Missing paired condition {baseline_name}")
                differences.append(
                    condition_values[lumber_name]["recall"][meeting_index]
                    - condition_values[baseline_name]["recall"][meeting_index]
                )
            meeting_effects.append(sum(differences) / len(differences))
        output[baseline] = {
            "environment_count": len(lumber),
            "recall": bootstrap.mean(meeting_effects),
        }
    return output


def lumber_targets(lumber_dir, meeting_ids):
    targets = set()
    for meeting_id in meeting_ids:
        artifact = read_json_object(lumber_dir / f"{meeting_id}.json")
        target = artifact.get("provenance", {}).get("config", {}).get("target_tokens")
        targets.add(target)
    return sorted(targets, key=lambda value: (value is None, str(value)))


def interval(estimate, signed=False):
    low, high = estimate["ci95"]
    pattern = "+.3f" if signed else ".3f"
    return (
        f"{format(estimate['mean'], pattern)} "
        f"[{format(low, pattern)}, {format(high, pattern)}]"
    )


def make_report(result):
    lines = [
        "# Ablation uncertainty analysis",
        "",
        f"**Scope:** {result['meeting_count']} meetings; "
        f"{result['bootstrap_samples']:,} paired cluster-bootstrap samples; "
        f"seed {result['seed']}.",
        "",
    ]
    if not result["configuration"]["matches_preset"]:
        lines += [
            "> **Configuration warning:** saved segmentation target(s) "
            f"{result['configuration']['artifact_lumber_targets']} do not match "
            f"the preset target {result['configuration']['preset_lumber_target']}. "
            "The estimates below describe the saved artifacts.",
            "",
        ]
    lines += [
        "Meetings, not questions, are resampled. Thus questions from the same meeting remain together. Point estimates are meeting-macro means; brackets are percentile 95% confidence intervals.",
        "",
        "## Headline average Lumber effect",
        "",
        "For each meeting and baseline, Lumber-minus-baseline recall is calculated separately in every retriever x evidence-budget environment and then averaged across those environments. These meeting-level average effects are finally averaged and cluster-bootstrapped across meetings.",
        "",
    ]
    for baseline, row in result["average_lumber_effects"].items():
        estimate = row["recall"]
        low, high = estimate["ci95"]
        verb = "increased" if estimate["mean"] >= 0 else "decreased"
        lines.append(
            f"- Across the {row['environment_count']} prespecified retrieval "
            f"environments, Lumber {verb} evidence recall by "
            f"{abs(estimate['mean']) * 100:.1f} percentage points relative to "
            f"{baseline}, 95% CI [{low * 100:+.1f}, {high * 100:+.1f}]."
        )
    lines += [
        "",
        "## All retrieval conditions",
        "",
        "| Condition | Recall [95% CI] | MRR [95% CI] | ROUGE-L [95% CI] | BERTScore F1 [95% CI] | Judge [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["conditions"].items():
        estimates = row["estimates"]
        lines.append(
            f"| {name} | {interval(estimates['recall'])} | "
            f"{interval(estimates['mrr'])} | {interval(estimates['rougeL'])} | "
            f"{interval(estimates['bertscore_f1'])} | "
            f"{interval(estimates['judge'])} |"
        )

    lines += [
        "",
        "## Paired Lumber contrasts",
        "",
        "Each value is Lumber minus the named deterministic chunker under the same retriever and evidence budget.",
        "",
        "| Comparison | dRecall [95% CI] | dROUGE-L [95% CI] | dBERTScore F1 [95% CI] | dJudge [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, estimates in result["lumber_contrasts"].items():
        lines.append(
            f"| {name} | {interval(estimates['recall'], True)} | "
            f"{interval(estimates['rougeL'], True)} | "
            f"{interval(estimates['bertscore_f1'], True)} | "
            f"{interval(estimates['judge'], True)} |"
        )

    lines += [
        "",
        "## Oracle answer models",
        "",
        "| Stage | ROUGE-L [95% CI] | BERTScore F1 [95% CI] | Judge [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["oracle_models"].items():
        estimates = row["estimates"]
        lines.append(
            f"| {name} | {interval(estimates['rougeL'])} | "
            f"{interval(estimates['bertscore_f1'])} | "
            f"{interval(estimates['judge'])} |"
        )

    lines += [
        "",
        "### Paired oracle-model differences",
        "",
        "| Comparison | dROUGE-L [95% CI] | dBERTScore F1 [95% CI] | dJudge [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for name, estimates in result["oracle_contrasts"].items():
        lines.append(
            f"| {name} | {interval(estimates['rougeL'], True)} | "
            f"{interval(estimates['bertscore_f1'], True)} | "
            f"{interval(estimates['judge'], True)} |"
        )

    gap = result.get("oracle_gap")
    if gap:
        lines += [
            "",
            "## Oracle evidence gap",
            "",
            f"Paired difference: **{gap['oracle_stage']} minus "
            f"{gap['retrieval_stage']} / {gap['condition']}**. Both stages use "
            "the same answer model, so this isolates the downstream association with evidence source.",
            "",
            "| dROUGE-L [95% CI] | dBERTScore F1 [95% CI] | dJudge [95% CI] |",
            "|---:|---:|---:|",
            f"| {interval(gap['estimates']['rougeL'], True)} | "
            f"{interval(gap['estimates']['bertscore_f1'], True)} | "
            f"{interval(gap['estimates']['judge'], True)} |",
        ]

    lines += [
        "",
        "## Interpretation limits",
        "",
        "An interval excluding zero is evidence that the paired meeting-level difference is consistently directional under this dataset sample. It is not a correction for the many exploratory comparisons, and only 20 meeting clusters are available. Judge and BERTScore uncertainty also does not include uncertainty from changing the evaluator model or prompt.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-condition", default="lumber__dense__w1024")
    args = parser.parse_args()

    run = load_run_config(args.preset)
    meeting_ids = run.meeting_ids()
    bootstrap = Bootstrap(len(meeting_ids), args.samples, args.seed)
    retrieval = read_json_object(run.retrieval_dir / "summary.json")
    evaluation = read_json_object(run.evaluation_dir / "summary.json")
    require_meetings(retrieval, meeting_ids, "Retrieval summary")

    answer_summaries = {
        stage.name: read_json_object(run.answers_dir / stage.name / "summary.json")
        for stage in run.answers
    }
    for name, summary in answer_summaries.items():
        require_meetings(summary, meeting_ids, f"Answer summary {name}")
        if name not in evaluation["stages"]:
            raise ValueError(f"Evaluation summary has no stage {name}")

    retrieval_stage = next(stage for stage in run.answers if stage.source == "retrieval")
    retrieval_answer = answer_summaries[retrieval_stage.name]
    retrieval_evaluation = evaluation["stages"][retrieval_stage.name]
    condition_values = {}
    conditions = {}
    for condition, spec in retrieval["configurations"].items():
        values = {metric: [] for metric in RETRIEVAL_METRICS}
        for meeting_id in meeting_ids:
            row = retrieval["per_meeting"][meeting_id][condition]
            for metric, field in RETRIEVAL_METRICS.items():
                values[metric].append(row[field])
        values.update(
            stage_values(
                retrieval_answer, retrieval_evaluation, meeting_ids, condition
            )
        )
        condition_values[condition] = values
        conditions[condition] = {
            "specification": spec,
            "estimates": estimate_metrics(values, bootstrap),
        }

    lumber_contrasts = {}
    for condition, spec in retrieval["configurations"].items():
        if spec["chunker"] != "lumber":
            continue
        for baseline in BASELINE_CHUNKERS:
            baseline_name = condition.replace("lumber__", f"{baseline}__", 1)
            if baseline_name not in condition_values:
                continue
            name = f"{condition} minus {baseline}"
            lumber_contrasts[name] = contrast_metrics(
                condition_values[condition],
                condition_values[baseline_name],
                bootstrap,
            )

    average_effects = average_lumber_effects(
        condition_values, retrieval["configurations"], bootstrap
    )

    oracle_values = {}
    oracle_models = {}
    oracle_stages = [stage for stage in run.answers if stage.source == "oracle"]
    for stage in oracle_stages:
        values = stage_values(
            answer_summaries[stage.name],
            evaluation["stages"][stage.name],
            meeting_ids,
            "oracle",
        )
        oracle_values[stage.name] = values
        oracle_models[stage.name] = {
            "model": stage.model.tag,
            "estimates": estimate_metrics(values, bootstrap),
        }

    oracle_contrasts = {}
    for earlier, later in combinations(oracle_stages, 2):
        name = f"{later.name} minus {earlier.name}"
        oracle_contrasts[name] = contrast_metrics(
            oracle_values[later.name],
            oracle_values[earlier.name],
            bootstrap,
        )

    oracle_gap = None
    matching_oracle = next(
        (stage for stage in oracle_stages if stage.model.tag == retrieval_stage.model.tag),
        None,
    )
    if matching_oracle and args.primary_condition in condition_values:
        oracle_gap = {
            "oracle_stage": matching_oracle.name,
            "retrieval_stage": retrieval_stage.name,
            "condition": args.primary_condition,
            "estimates": contrast_metrics(
                oracle_values[matching_oracle.name],
                condition_values[args.primary_condition],
                bootstrap,
            ),
        }

    artifact_targets = lumber_targets(run.lumber_dir, meeting_ids)
    result = {
        "preset": str(args.preset),
        "meeting_ids": meeting_ids,
        "meeting_count": len(meeting_ids),
        "bootstrap_unit": "meeting",
        "bootstrap_samples": args.samples,
        "interval": "percentile 95%",
        "seed": args.seed,
        "configuration": {
            "preset_lumber_target": run.segmentation.target_tokens,
            "artifact_lumber_targets": artifact_targets,
            "matches_preset": artifact_targets == [run.segmentation.target_tokens],
        },
        "conditions": conditions,
        "average_lumber_effects": average_effects,
        "lumber_contrasts": lumber_contrasts,
        "oracle_models": oracle_models,
        "oracle_contrasts": oracle_contrasts,
        "oracle_gap": oracle_gap,
    }
    write_json(args.output, result)
    report = args.output.with_suffix(".md")
    report.write_text(make_report(result), encoding="utf-8")
    print(f"Uncertainty data: {args.output}")
    print(f"Uncertainty report: {report}")


if __name__ == "__main__":
    main()
