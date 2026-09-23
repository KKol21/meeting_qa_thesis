# Plot generation and data calculations

This document describes how the eight result figures are produced, where their
numbers come from, and which aggregation choices must be considered when
interpreting them. The figure index itself is in
[`RESULT_VISUALIZATIONS.md`](RESULT_VISUALIZATIONS.md).

## Data flow

The plotting script does not rerun retrieval, answer generation, or evaluation.
It only reads derived JSON artifacts and renders them:

```text
meeting-level experiment artifacts
  -> stage summaries and secondary-analysis JSON
  -> src/tools/plot_results.py
  -> docs/figures/results/*.png and *.pdf
```

The eight plots use the following inputs.

| Figures | Input artifact | Artifact-producing code |
|---|---|---|
| 01, 02, 03, 05 | `runs/ablations/full/uncertainty.json` | [`analyze_uncertainty.py`](../src/tools/analyze_uncertainty.py) |
| 04 | `runs/ablations/full/budget-deltas.json` | [`analyze_budget_deltas.py`](../src/tools/analyze_budget_deltas.py) |
| 06 | `runs/ablations/baseline-sweep/sweep.json` | [`sweep_baseline_sizes.py`](../src/tools/sweep_baseline_sizes.py) |
| 07 | `runs/ablations/boundary-control/control.json` | [`boundary_shuffled_control.py`](../src/tools/boundary_shuffled_control.py) |
| 08 | `runs/ablations/validation-full/clipping-sensitivity.json` | [`analyze_clipping_sensitivity.py`](../src/tools/analyze_clipping_sensitivity.py) |

Figures 01--05 use the held-out QMSum test set: 35 meetings and 244
questions. Figures 06--08 are validation analyses based on the fixed
20-meeting, 142-question subset.

## Shared metric definitions

### Retrieved evidence

For each question, chunks are ranked by dense, BM25, or hybrid retrieval.
[`select_evidence()`](../src/meeting_qa_chunking/evidence.py) visits chunks in
rank order, removes duplicate source-word positions (a safety measure which is not used as none of the examined chunking methods create overlapping chunks), and takes words until the
configured budget is exactly filled. If necessary, the final chunk or turn is
clipped. Ranking is therefore exhaustive, but only the highest-ranked evidence
that fits the budget is selected.

Let (R_q) be the retrieved source words and (G_q) the words contained in the
union of QMSum's annotated relevant turns. Then:

```text
precision = |R_q intersect G_q| / |R_q|
recall    = |R_q intersect G_q| / |G_q|
F1        = 2 * precision * recall / (precision + recall)
```

QMSum relevance is annotated at turn level. Consequently, every word in an
annotated turn is treated as relevant and every word outside those turns as
non-relevant. Speaker labels and turn identifiers do not count toward the word
budget or these metrics.

### Answer metrics

- **ROUGE-L** is the stemmed ROUGE-L F-measure between the generated and
  reference answers, calculated with `rouge-score`.
- **BERTScore F1** compares the generated and reference answers using
  `roberta-large`, layer 17, with the package baseline rescaling enabled.
- **LLM judge** is an ordinal score from 1 to 3. The judge receives the
  question, reference answer, annotated gold evidence, and candidate answer.
  Its scale is incorrect/invalid, partially correct, and correct. It does not
  receive the retrieved evidence used to generate the answer, so this score is
  an answer-quality measure rather than a direct faithfulness measure.

### Meeting-macro aggregation

Unless a figure says otherwise, questions are first averaged within each
meeting and the meeting means are then averaged. Every meeting therefore has
equal weight regardless of its number of questions.

Uncertainty estimates use 10,000 percentile bootstrap samples with seed 42.
Whole meetings are sampled with replacement, retaining all within-meeting
questions and all paired conditions. A plotted 95% interval is the 2.5th to
97.5th percentile of the resulting estimates. The helper implementation is in
[`uncertainty.py`](../src/meeting_qa_chunking/uncertainty.py).

## Figure calculations

### 01 — Test retrieval trade-off

For each of the 27 chunker × retriever × budget conditions,
`analyze_uncertainty.py` reads the retrieval stage's per-meeting precision and
recall. It calculates the meeting-macro mean and a meeting-cluster bootstrap
interval for each metric.

[`plot_retrieval_tradeoff()`](../src/tools/plot_results.py) creates one panel
per retriever. The horizontal coordinate is mean precision and the vertical
coordinate is mean recall. Horizontal and vertical error bars are their
respective 95% intervals. Within a chunker, the three connected points progress
from 512 to 1,024 to 2,048 evidence words. The connecting line only shows the
budget sequence; it does not imply interpolation between budgets.

### 02 — Test downstream performance

The answer and evaluation stages first calculate ROUGE-L, BERTScore F1, and
judge means per condition within each meeting. `analyze_uncertainty.py` then
calculates the equally weighted meeting mean and bootstrap interval.

[`plot_downstream()`](../src/tools/plot_results.py) places retrievers in columns
and answer metrics in rows. Each line represents one chunker across the three
evidence budgets. Error bars are intervals for each condition mean, not for a
paired difference between budgets or chunkers. Overlap between two such
intervals is therefore not a formal paired comparison.

### 03 — Paired Lumber effects

For each retriever × budget environment, the analysis subtracts the baseline
meeting mean from the Lumber meeting mean. This is done separately for
turn-packed and word-packed chunks and for recall, ROUGE-L, BERTScore F1, and
judge score. The paired meeting differences are then averaged and bootstrapped.

The headline recall effect uses an additional within-meeting step. For each
meeting, the nine Lumber-minus-baseline recall differences are averaged across
the three retrievers and three budgets. These 35 meeting-level averages are
then averaged and bootstrapped. Thus, the nine environments do not count as
independent observations.

[`plot_lumber_effects()`](../src/tools/plot_results.py) shows the 18 cell-level
contrasts around a zero-effect reference line. The subtitle reports the two
headline average recall effects.

### 04 — Retrieval changes versus answer changes

[`analyze_budget_deltas.py`](../src/tools/analyze_budget_deltas.py) joins the
question-level retrieval, answer, and evaluation records for the same question
and condition. Every delta is calculated as the 2,048-word value minus the
1,024-word value. Precision loss is defined in the opposite direction:

```text
recall gain    = recall_2048 - recall_1024
precision loss = precision_1024 - precision_2048
answer change  = score_2048 - score_1024
```

The plotted analysis is restricted to Lumber with dense retrieval. Blue points
are all 244 question-level pairs. Judge-score points are vertically jittered
with seed 42 for legibility; the original ordinal differences are used in every
calculation.

Spearman correlations are computed over the pooled questions. Their intervals
cluster-bootstrap meetings: when a meeting is sampled, all its questions are
included before the correlation is recomputed.

The orange markers use fixed diagnostic bins:

- recall gain: below 0.10, 0.10--0.25, and at least 0.25;
- precision loss: below 0.05, 0.05--0.15, and at least 0.15.

Within each bin, question deltas are averaged within each represented meeting,
then meetings are averaged equally. Their vertical error bars are meeting-level
bootstrap intervals for the answer change. These bins are descriptive and were
not optimized to maximize a result.

### 05 — Oracle evidence and answer models

`analyze_uncertainty.py` combines one retrieved-evidence condition with three
oracle answer stages:

- retrieved evidence: Lumber + dense retrieval + 1,024 words, Qwen2.5-14B;
- uncapped annotated evidence: Qwen2.5-7B;
- uncapped annotated evidence: Qwen2.5-14B;
- uncapped annotated evidence: 4-bit Qwen2.5-32B.

Each point is the meeting-macro mean with a meeting-cluster bootstrap interval.
The annotation above each panel is a paired contrast between oracle-14B and
retrieved-14B, calculated within meetings before bootstrapping. This contrast
holds the answer model fixed, but changes both evidence content and context
length because oracle evidence is not capped at 1,024 words.

### 06 — Deterministic chunk-size sensitivity

[`sweep_baseline_sizes.py`](../src/tools/sweep_baseline_sizes.py) reconstructs
turn-packed and word-packed chunks with 128-, 256-, and 512-word targets. It
reruns all three retrievers and all three evidence budgets on the validation
subset, using the same evidence-selection and retrieval metrics as the main
experiment.

For each chunker, metric, retriever, and budget, the plotted cell is:

```text
meeting-macro metric at tested size - meeting-macro metric at 256 words
```

The heatmaps show recall and F1. The diverging colour scale is shared across
all four panels and centred at zero. This figure is descriptive and does not
show confidence intervals.

### 07 — Boundary-shuffled Lumber control

[`boundary_shuffled_control.py`](../src/tools/boundary_shuffled_control.py)
starts with the selected 1,000-target Lumber segmentation for every validation
meeting. Each control permutes Lumber's exact turns-per-chunk lengths across
legal turn boundaries. For each of ten random versions, 1,000 candidate
permutations are considered and the candidate whose sorted word-length
distribution is closest to Lumber's is retained. A candidate identical to the
actual boundary set is rejected.

Retrieval is run independently on actual Lumber and every control partition.
For each meeting and environment:

1. questions are averaged for actual Lumber;
2. control results are averaged across questions and random versions;
3. the control mean is subtracted from actual Lumber;
4. the resulting 20 paired meeting differences are averaged and bootstrapped.

[`plot_boundary_control()`](../src/tools/plot_results.py) reports recall and F1
differences with 95% intervals. Positive values favour the actual Lumber
boundaries. The control exactly preserves the chunk count and turns-per-chunk
multiset, but matches the word-length distribution only approximately.

### 08 — Final-chunk clipping sensitivity

[`analyze_clipping_sensitivity.py`](../src/tools/analyze_clipping_sensitivity.py)
reuses saved validation rankings and chunk boundaries. It reconstructs three
policies:

- **clip:** take exactly the word budget, clipping the final chunk if needed;
- **drop partial:** omit the partially fitting final chunk and leave the
  remaining budget unused;
- **expand partial:** include that final chunk in full, exceeding the nominal
  budget when necessary.

Drop does not backfill with a lower-ranked chunk. For each policy, questions
are averaged within meetings and meetings receive equal weight. Every heatmap
cell is the policy's meeting-macro recall or F1 minus the corresponding clip
value. This retrieval-only figure is descriptive and does not show confidence
intervals.

## Reproduction

The derived test artifacts can be rebuilt with:

```powershell
$env:PYTHONPATH = "src"
python src/tools/analyze_uncertainty.py `
  --preset src/configs/ablation-full.toml `
  --output runs/ablations/full/uncertainty.json `
  --samples 10000 --seed 42

python src/tools/analyze_budget_deltas.py `
  --preset src/configs/ablation-full.toml `
  --output runs/ablations/full/budget-deltas.json `
  --samples 10000 --seed 42
```

The validation analyses can be rebuilt with:

```powershell
python src/tools/sweep_baseline_sizes.py `
  --preset src/configs/ablation-validation-full.toml `
  --sizes 128 256 512 `
  --output runs/ablations/baseline-sweep/sweep.json

python src/tools/boundary_shuffled_control.py `
  --preset src/configs/ablation-validation-full.toml `
  --segmentation-dir runs/ablations/lumber-sweep/segmentation/1000 `
  --target 1000 --randomizations 10 --candidates 1000 --seed 42 `
  --output runs/ablations/boundary-control/control.json

python src/tools/analyze_clipping_sensitivity.py `
  --preset src/configs/ablation-validation-full.toml `
  --output runs/ablations/validation-full/clipping-sensitivity.json
```

The baseline sweep and boundary control rerun retrieval and therefore require
the embedding model. The other analysis commands operate on already saved
artifacts.

Finally, generate all eight figures:

```powershell
pip install -e ".[plots]"
python src/tools/plot_results.py
```

[`plot_results.py`](../src/tools/plot_results.py) writes a 220-DPI PNG for
review and a vector PDF for typesetting. PDF creation and modification dates
are omitted so repeated runs over unchanged data produce identical files.

## Interpretation cautions

- Cell-level intervals are exploratory and are not adjusted for multiple
  comparisons.
- An interval for a condition mean is not interchangeable with an interval for
  a paired condition difference.
- Retrieval precision and recall inherit the limitations of QMSum's turn-level
  evidence annotations.
- ROUGE-L and BERTScore compare against one reference answer and can understate
  semantically valid alternatives.
- The 1--3 judge score is ordinal even though its meeting means and paired
  differences are summarized numerically.
- Figures 04, 06, 07, and 08 are diagnostic or sensitivity analyses rather than
  additional independent confirmatory tests.
