"""Generate the thesis result figures from saved JSON artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


CHUNKERS = ("turn_packed", "word_packed", "lumber")
RETRIEVERS = ("dense", "bm25", "hybrid")
BUDGETS = (512, 1024, 2048)
COLORS = {
    "turn_packed": "#0072B2",
    "word_packed": "#E69F00",
    "lumber": "#009E73",
}
LABELS = {
    "turn_packed": "Turn-packed",
    "word_packed": "Word-packed",
    "lumber": "Lumber",
    "dense": "Dense",
    "bm25": "BM25",
    "hybrid": "Hybrid",
}
MARKERS = {"turn_packed": "o", "word_packed": "s", "lumber": "^"}


def read(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def interval(record: dict, metric: str) -> tuple[float, float, float]:
    estimate = record[metric]
    return estimate["mean"], estimate["ci95"][0], estimate["ci95"][1]


def errorbar(axis, x, values, **kwargs) -> None:
    means = [value[0] for value in values]
    errors = [
        [value[0] - value[1] for value in values],
        [value[2] - value[0] for value in values],
    ]
    axis.errorbar(x, means, yerr=errors, capsize=2, **kwargs)


def finish(figure, output_dir: Path, name: str) -> None:
    figure.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def condition(uncertainty: dict, chunker: str, retriever: str, budget: int) -> dict:
    return uncertainty["conditions"][
        f"{chunker}__{retriever}__w{budget}"
    ]["estimates"]


def plot_retrieval_tradeoff(uncertainty: dict, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharex=True, sharey=True)
    for axis, retriever in zip(axes, RETRIEVERS):
        for chunker in CHUNKERS:
            rows = [condition(uncertainty, chunker, retriever, b) for b in BUDGETS]
            precision = [row["precision"]["mean"] for row in rows]
            recall = [row["recall"]["mean"] for row in rows]
            xerr = [
                [row["precision"]["mean"] - row["precision"]["ci95"][0] for row in rows],
                [row["precision"]["ci95"][1] - row["precision"]["mean"] for row in rows],
            ]
            yerr = [
                [row["recall"]["mean"] - row["recall"]["ci95"][0] for row in rows],
                [row["recall"]["ci95"][1] - row["recall"]["mean"] for row in rows],
            ]
            axis.errorbar(
                precision,
                recall,
                xerr=xerr,
                yerr=yerr,
                color=COLORS[chunker],
                marker=MARKERS[chunker],
                linewidth=1.5,
                markersize=5,
                capsize=2,
                label=LABELS[chunker],
            )
        axis.set_title(LABELS[retriever])
        axis.set_xlabel("Evidence precision")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Evidence recall")
    axes[0].legend(frameon=False, loc="upper right")
    figure.suptitle("Test-set retrieval trade-off by evidence budget")
    figure.text(
        0.5, 0.015,
        "Along each line: 512 → 1,024 → 2,048 evidence words (right to left)",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    finish(figure, output_dir, "01-test-retrieval-tradeoff")


def plot_downstream(uncertainty: dict, output_dir: Path) -> None:
    metrics = (
        ("rougeL", "ROUGE-L"),
        ("bertscore_f1", "BERTScore F1"),
        ("judge", "LLM judge (1–3)"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True, sharey="row")
    x = np.arange(len(BUDGETS))
    for row, (metric, label) in enumerate(metrics):
        for column, retriever in enumerate(RETRIEVERS):
            axis = axes[row, column]
            for chunker in CHUNKERS:
                values = [
                    interval(condition(uncertainty, chunker, retriever, b), metric)
                    for b in BUDGETS
                ]
                errorbar(
                    axis,
                    x,
                    values,
                    color=COLORS[chunker],
                    marker=MARKERS[chunker],
                    linewidth=1.4,
                    markersize=4,
                    label=LABELS[chunker],
                )
            if row == 0:
                axis.set_title(LABELS[retriever])
            if column == 0:
                axis.set_ylabel(label)
            axis.set_xticks(x, ["512", "1,024", "2,048"])
            axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Evidence budget (words)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Test-set answer quality changes little across chunkers", y=0.995)
    figure.legend(
        handles, labels, frameon=False, ncol=3, loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.925))
    finish(figure, output_dir, "02-test-downstream-performance")


def plot_lumber_effects(uncertainty: dict, output_dir: Path) -> None:
    metrics = (
        ("recall", "Δ recall"),
        ("rougeL", "Δ ROUGE-L"),
        ("bertscore_f1", "Δ BERTScore F1"),
        ("judge", "Δ judge"),
    )
    environments = [(r, b) for r in RETRIEVERS for b in BUDGETS]
    baselines = ("turn_packed", "word_packed")
    figure, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)
    y = np.arange(len(environments))
    for axis, (metric, label) in zip(axes, metrics):
        for offset, baseline in zip((-0.12, 0.12), baselines):
            values = []
            for retriever, budget in environments:
                key = f"lumber__{retriever}__w{budget} minus {baseline}"
                values.append(interval(uncertainty["lumber_contrasts"][key], metric))
            means = [value[0] for value in values]
            errors = [
                [value[0] - value[1] for value in values],
                [value[2] - value[0] for value in values],
            ]
            axis.errorbar(
                means,
                y + offset,
                xerr=errors,
                fmt=MARKERS[baseline],
                color=COLORS[baseline],
                capsize=2,
                markersize=4,
                label=f"vs {LABELS[baseline]}",
            )
        axis.axvline(0, color="0.35", linewidth=1)
        axis.set_xlabel(label)
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(
        y, [f"{LABELS[r]} · {b:,}" for r, b in environments]
    )
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    headline = []
    for baseline in baselines:
        value = uncertainty["average_lumber_effects"][baseline]["recall"]
        headline.append(
            f"vs {LABELS[baseline]} {value['mean'] * 100:+.1f} pp "
            f"[{value['ci95'][0] * 100:+.1f}, {value['ci95'][1] * 100:+.1f}]"
        )
    figure.suptitle("Paired Lumber effects within matched test environments", y=0.99)
    figure.text(
        0.5, 0.945, "Average recall effect across 9 environments: " + "; ".join(headline),
        ha="center", fontsize=9,
    )
    figure.legend(
        handles, labels, frameon=False, ncol=2, loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.91))
    finish(figure, output_dir, "03-test-lumber-paired-effects")


def plot_budget_proportionality(budget_data: dict, output_dir: Path) -> None:
    rows = [
        row
        for row in budget_data["question_deltas"]
        if row["chunker"] == "lumber" and row["retriever"] == "dense"
    ]
    outcomes = (
        ("rougeL", "Δ ROUGE-L"),
        ("bertscore_f1", "Δ BERTScore F1"),
        ("judge", "Δ judge"),
    )
    predictors = (
        ("recall_gain", "Recall gain", "recall_gain_bins"),
        ("precision_loss", "Precision loss", "precision_loss_bins"),
    )
    bin_markers = {"small": "D", "moderate": "s", "large": "^"}
    rng = np.random.default_rng(42)
    figure, axes = plt.subplots(3, 2, figsize=(7.2, 8), sharex="col")
    for column, (predictor, predictor_label, bin_name) in enumerate(predictors):
        for row_index, (metric, metric_label) in enumerate(outcomes):
            axis = axes[row_index, column]
            x = np.array(
                [
                    item["delta"]["recall"]
                    if predictor == "recall_gain"
                    else item["precision_loss"]
                    for item in rows
                ]
            )
            y = np.array([item["delta"][metric] for item in rows])
            displayed_y = y + rng.uniform(-0.045, 0.045, len(y)) if metric == "judge" else y
            axis.scatter(x, displayed_y, s=12, alpha=0.25, color="#0072B2", linewidths=0)
            for group in budget_data["focus"][bin_name]:
                estimates = group["estimates"]
                group_x = (
                    estimates["recall"]["mean"]
                    if predictor == "recall_gain"
                    else -estimates["precision"]["mean"]
                )
                group_y, low, high = interval(estimates, metric)
                axis.errorbar(
                    group_x,
                    group_y,
                    yerr=[[group_y - low], [high - group_y]],
                    fmt=bin_markers[group["name"]],
                    color="#D55E00",
                    markersize=5,
                    capsize=3,
                )
            correlation = budget_data["focus"]["correlations"][
                f"{predictor}__{metric}"
            ]
            low, high = correlation["ci95"]
            axis.text(
                0.03,
                0.97,
                f"Spearman ρ = {correlation['spearman']:.2f} [{low:.2f}, {high:.2f}]",
                transform=axis.transAxes,
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
            )
            axis.axhline(0, color="0.45", linewidth=0.8)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(predictor_label)
            if column == 0:
                axis.set_ylabel(metric_label)
            if row_index == len(outcomes) - 1:
                axis.set_xlabel(predictor_label)
    figure.suptitle("Associations between retrieval and answer changes", y=0.995)
    figure.text(0.5, 0.964, "Lumber + dense retrieval; budget 1,024 → 2,048 words", ha="center", fontsize=9)
    handles = [
        Line2D([0], [0], marker=marker, color="#D55E00", linestyle="none", markersize=6, label=name.title())
        for name, marker in bin_markers.items()
    ]
    figure.legend(
        handles=handles, title="Bin means ± 95% CI", frameon=False, ncol=3,
        loc="upper center", bbox_to_anchor=(0.5, 0.95), fontsize=8.5, title_fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    finish(figure, output_dir, "04-test-budget-proportionality")


def plot_oracle_models(uncertainty: dict, output_dir: Path) -> None:
    metrics = (
        ("rougeL", "ROUGE-L"),
        ("bertscore_f1", "BERTScore F1"),
        ("judge", "LLM judge (1–3)"),
    )
    retrieval_key = uncertainty["oracle_gap"]["condition"]
    sources = [
        ("Ret.\n14B", uncertainty["conditions"][retrieval_key]["estimates"], "0.45"),
        ("O.\n7B", uncertainty["oracle_models"]["oracle-7b"]["estimates"], "#56B4E9"),
        ("O.\n14B", uncertainty["oracle_models"]["oracle-14b"]["estimates"], "#0072B2"),
        ("O.\n32B", uncertainty["oracle_models"]["oracle-32b-bnb4"]["estimates"], "#CC79A7"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.7))
    x = np.arange(len(sources))
    for axis, (metric, label) in zip(axes, metrics):
        values = [interval(estimates, metric) for _, estimates, _ in sources]
        for index, ((_, _, color), value) in enumerate(zip(sources, values)):
            axis.errorbar(
                index,
                value[0],
                yerr=[[value[0] - value[1]], [value[2] - value[0]]],
                fmt="o",
                color=color,
                capsize=3,
                markersize=6,
            )
        axis.set_xticks(x, [item[0] for item in sources])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        gap = uncertainty["oracle_gap"]["estimates"][metric]
        axis.set_title(
            f"14B paired gap: {gap['mean']:+.3f}\n"
            f"95% CI [{gap['ci95'][0]:+.3f}, {gap['ci95'][1]:+.3f}]",
            fontsize=8.5,
            loc="left",
            pad=9,
        )
    figure.suptitle("Answer quality by evidence condition and model", y=0.995)
    figure.text(
        0.5, 0.91,
        "Ret. = 1,024-word retrieved; O. = uncapped annotated oracle; 32B is 4-bit",
        ha="center", fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    finish(figure, output_dir, "05-test-oracle-models")


def plot_lumber_target_sweep(data: dict, output_dir: Path) -> None:
    targets = data["targets"]
    diagnostics = [data["selection"]["diagnostics"][str(target)] for target in targets]
    selected = data["selection"]["recommended_target"]
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    recall = [row["mean_recall_across_conditions"] for row in diagnostics]
    f1 = [row["mean_f1_across_conditions"] for row in diagnostics]
    axes[0].plot(targets, recall, "o-", label="Recall", color="#0072B2")
    axes[0].plot(targets, f1, "s-", label="F1", color="#D55E00")
    axes[0].axhspan(max(recall) - 0.01, max(recall), color="0.85", zorder=0)
    axes[0].set_ylabel("Mean across 9 environments")
    axes[0].legend(frameon=False)

    axes[1].plot(
        targets,
        [row["mean_zero_hit_across_conditions"] for row in diagnostics],
        "o-",
        color="#CC79A7",
    )
    axes[1].set_ylabel("Zero-hit rate")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))

    axes[2].plot(
        targets,
        [row["median_chunk_words"] for row in diagnostics],
        "o-",
        color="#009E73",
        label="Lumber median",
    )
    for chunker in ("turn_packed", "word_packed"):
        median_words = data["baselines"][chunker]["words_per_chunk"]["median"]
        axes[2].axhline(
            median_words,
            color=COLORS[chunker],
            linestyle="--",
            label=f"{LABELS[chunker]} median",
        )
    axes[2].set_ylabel("Median chunk length (words)")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.axvline(selected, color="0.25", linestyle=":")
        axis.set_xlabel("Lumber target (pseudo-tokens)")
        axis.grid(alpha=0.2)
    figure.suptitle("Validation selection of the Lumber target")
    figure.tight_layout()
    finish(figure, output_dir, "06-validation-lumber-target")


def plot_baseline_sizes(data: dict, output_dir: Path) -> None:
    chunkers = ("turn_packed", "word_packed")
    metrics = (("recall", "Recall"), ("f1", "F1"))
    environments = [(r, b) for r in RETRIEVERS for b in BUDGETS]
    sizes = data["sizes"]

    def value(chunker, size, retriever, budget, metric):
        name = f"{chunker}__s{size}__{retriever}__w{budget}"
        reference = f"{chunker}__s256__{retriever}__w{budget}"
        return data["aggregate"][name][metric] - data["aggregate"][reference][metric]

    matrices = {
        (chunker, metric): np.array(
            [
                [value(chunker, size, retriever, budget, metric) for size in sizes]
                for retriever, budget in environments
            ]
        )
        for chunker in chunkers
        for metric, _label in metrics
    }
    limit = max(abs(matrix).max() for matrix in matrices.values())
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    image = None
    for row, chunker in enumerate(chunkers):
        for column, (metric, label) in enumerate(metrics):
            axis = axes[row, column]
            matrix = matrices[(chunker, metric)]
            image = axis.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
            for y in range(len(environments)):
                for x in range(len(sizes)):
                    axis.text(x, y, f"{matrix[y, x]:+.3f}", ha="center", va="center", fontsize=8)
            axis.set_xticks(range(len(sizes)), sizes)
            axis.set_yticks(
                range(len(environments)),
                [f"{LABELS[r]} · {b:,}" for r, b in environments],
            )
            axis.set_xlabel("Chunk-size target (words)")
            axis.set_title(f"{LABELS[chunker]}: Δ {label}")
    figure.colorbar(image, ax=axes, shrink=0.8, label="Difference from the 256-word condition")
    figure.suptitle("Validation sensitivity of deterministic chunk size")
    finish(figure, output_dir, "07-validation-baseline-size")


def plot_boundary_control(data: dict, output_dir: Path) -> None:
    environments = [(r, b) for r in RETRIEVERS for b in BUDGETS]
    figure, axes = plt.subplots(1, 2, figsize=(9, 5.4), sharey=True)
    y = np.arange(len(environments))
    for axis, (metric, label) in zip(axes, (("recall", "Δ recall"), ("f1", "Δ F1"))):
        values = [
            interval(data["aggregate"][f"{r}__w{b}"]["actual_minus_control"], metric)
            for r, b in environments
        ]
        means = [value[0] for value in values]
        errors = [
            [value[0] - value[1] for value in values],
            [value[2] - value[0] for value in values],
        ]
        axis.errorbar(means, y, xerr=errors, fmt="o", color="#009E73", capsize=3)
        axis.axvline(0, color="0.35", linewidth=1)
        axis.set_xlabel(f"Actual Lumber minus shuffled control: {label}")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, [f"{LABELS[r]} · {b:,}" for r, b in environments])
    axes[0].invert_yaxis()
    figure.suptitle("Validation boundary-shuffled Lumber control")
    figure.tight_layout()
    finish(figure, output_dir, "08-validation-boundary-control")


def plot_clipping_sensitivity(data: dict, output_dir: Path) -> None:
    environments = [(r, b) for r in RETRIEVERS for b in BUDGETS]
    policies = (("drop_partial", "Drop partial"), ("expand_partial", "Expand partial"))
    metrics = (("recall", "Δ recall"), ("f1", "Δ F1"))
    matrices = {}
    for metric, _ in metrics:
        for policy, _ in policies:
            matrices[(metric, policy)] = np.array(
                [
                    [
                        data["aggregate"][f"{chunker}__{retriever}__w{budget}"][policy][metric]
                        - data["aggregate"][f"{chunker}__{retriever}__w{budget}"]["clip"][metric]
                        for chunker in CHUNKERS
                    ]
                    for retriever, budget in environments
                ]
            )

    figure = plt.figure(figsize=(9, 8), layout="constrained")
    grid = figure.add_gridspec(2, 3, width_ratios=(1, 1, 0.06))
    axes = np.array(
        [
            [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])],
            [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
        ]
    )
    for row, (metric, metric_label) in enumerate(metrics):
        limit = max(np.abs(matrices[(metric, policy)]).max() for policy, _ in policies)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        for column, (policy, policy_label) in enumerate(policies):
            axis = axes[row, column]
            matrix = matrices[(metric, policy)]
            image = axis.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
            for y, x in np.ndindex(matrix.shape):
                text_color = "white" if abs(matrix[y, x]) > 0.6 * limit else "black"
                axis.text(
                    x, y, f"{matrix[y, x]:+.3f}", ha="center", va="center",
                    fontsize=7, color=text_color,
                )
            axis.set_title(f"{policy_label}: {metric_label}")
        color_axis = figure.add_subplot(grid[row, 2])
        figure.colorbar(image, cax=color_axis, label=f"Policy minus clip: {metric_label}")
    environment_labels = [f"{LABELS[r]} · {b:,}" for r, b in environments]
    for row in range(2):
        for column in range(2):
            axis = axes[row, column]
            axis.set_xticks(
                np.arange(len(CHUNKERS)), [LABELS[c] for c in CHUNKERS],
                rotation=20,
            )
            axis.set_yticks(
                np.arange(len(environments)),
                environment_labels if column == 0 else [],
            )
    figure.suptitle("Validation sensitivity to final-chunk budget handling")
    finish(figure, output_dir, "09-validation-clipping-policy")


def plot_boundary_models(data: dict, output_dir: Path) -> None:
    models = data["models"]
    labels = [model.replace("qwen2.5-", "") for model in models]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))

    values = [interval(data["headline"][model], "recall") for model in models]
    errorbar(axes[0], np.arange(len(models)), values, fmt="o", color="#009E73")
    axes[0].set_xticks(np.arange(len(models)), labels, rotation=20)
    axes[0].set_ylabel("Mean recall across 9 environments")

    medians = [data["chunking"][model]["words_per_chunk"]["median"] for model in models]
    axes[1].bar(np.arange(len(models)), medians, color="#56B4E9")
    axes[1].set_xticks(np.arange(len(models)), labels, rotation=20)
    axes[1].set_ylabel("Median chunk length (words)")

    matrix = np.eye(len(models))
    for row in data["boundary_overlap"]:
        left, right = models.index(row["left"]), models.index(row["right"])
        matrix[left, right] = matrix[right, left] = row["jaccard"]
    image = axes[2].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for row, column in np.ndindex(matrix.shape):
        axes[2].text(column, row, f"{matrix[row, column]:.0%}", ha="center", va="center", fontsize=8)
    axes[2].set_xticks(np.arange(len(models)), labels, rotation=20)
    axes[2].set_yticks(np.arange(len(models)), labels)
    axes[2].set_title("Boundary Jaccard")
    figure.colorbar(image, ax=axes[2], shrink=0.75)
    for axis in axes[:2]:
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Five-meeting Lumber boundary-model diagnostic")
    figure.tight_layout()
    finish(figure, output_dir, "10-validation-boundary-model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, default=Path("runs/ablations/full"))
    parser.add_argument(
        "--validation", type=Path, default=Path("runs/ablations/validation-full")
    )
    parser.add_argument(
        "--lumber-sweep", type=Path,
        default=Path("runs/ablations/lumber-sweep/sweep.json"),
    )
    parser.add_argument(
        "--baseline-sweep", type=Path,
        default=Path("runs/ablations/baseline-sweep/sweep.json"),
    )
    parser.add_argument(
        "--boundary-control", type=Path,
        default=Path("runs/ablations/boundary-control/control.json"),
    )
    parser.add_argument(
        "--model-check", type=Path,
        default=Path("runs/ablations/lumber-model-check/check.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/figures/results")
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    uncertainty = read(args.full / "uncertainty.json")
    budget_data = read(args.full / "budget-deltas.json")
    plot_retrieval_tradeoff(uncertainty, args.output)
    plot_downstream(uncertainty, args.output)
    plot_lumber_effects(uncertainty, args.output)
    plot_budget_proportionality(budget_data, args.output)
    plot_oracle_models(uncertainty, args.output)
    plot_lumber_target_sweep(read(args.lumber_sweep), args.output)
    plot_baseline_sizes(read(args.baseline_sweep), args.output)
    plot_boundary_control(read(args.boundary_control), args.output)
    plot_clipping_sensitivity(
        read(args.validation / "clipping-sensitivity.json"), args.output
    )
    plot_boundary_models(read(args.model_check), args.output)
    print(f"Result figures: {args.output}")


if __name__ == "__main__":
    main()
