# Evidence-budget delta analysis

**Dataset:** QMSum test; 35 meetings, 244 questions.

**Comparison:** 2,048 minus 1,024 evidence words. Questions are first averaged within each meeting, and meetings receive equal weight. Brackets are percentile 95% meeting-cluster bootstrap intervals (10,000 samples; seed 42).

## Budget effect in every retrieval environment

| Chunker | Retriever | dPrecision | dRecall | dROUGE-L | dBERTScore F1 | dJudge |
|---|---|---:|---:|---:|---:|---:|
| turn_packed | dense | -0.068 [-0.083, -0.053] | +0.166 [+0.131, +0.201] | +0.004 [-0.001, +0.009] | +0.006 [-0.002, +0.015] | +0.054 [-0.014, +0.121] |
| turn_packed | bm25 | -0.059 [-0.072, -0.047] | +0.149 [+0.127, +0.172] | +0.004 [-0.002, +0.010] | +0.001 [-0.008, +0.011] | +0.045 [-0.032, +0.118] |
| turn_packed | hybrid | -0.080 [-0.095, -0.064] | +0.140 [+0.117, +0.165] | +0.004 [-0.001, +0.009] | +0.004 [-0.004, +0.013] | +0.056 [-0.002, +0.117] |
| word_packed | dense | -0.065 [-0.083, -0.047] | +0.186 [+0.154, +0.220] | +0.005 [-0.001, +0.011] | +0.007 [-0.002, +0.016] | +0.018 [-0.030, +0.065] |
| word_packed | bm25 | -0.065 [-0.082, -0.049] | +0.151 [+0.118, +0.189] | +0.003 [-0.002, +0.008] | +0.004 [-0.006, +0.014] | +0.066 [+0.005, +0.132] |
| word_packed | hybrid | -0.058 [-0.073, -0.042] | +0.180 [+0.146, +0.214] | +0.005 [-0.001, +0.010] | -0.000 [-0.009, +0.008] | +0.032 [-0.052, +0.114] |
| lumber | dense | -0.071 [-0.088, -0.054] | +0.192 [+0.161, +0.223] | +0.002 [-0.004, +0.009] | +0.002 [-0.006, +0.010] | +0.039 [-0.021, +0.098] |
| lumber | bm25 | -0.064 [-0.081, -0.047] | +0.166 [+0.135, +0.199] | +0.010 [+0.003, +0.015] | +0.010 [-0.000, +0.019] | +0.101 [+0.029, +0.174] |
| lumber | hybrid | -0.077 [-0.095, -0.059] | +0.149 [+0.122, +0.176] | +0.006 [+0.001, +0.012] | +0.002 [-0.007, +0.012] | +0.082 [+0.021, +0.143] |

## Focused proportionality check: lumber + dense

The bins were fixed before inspecting these summaries. They are descriptive, not optimized cut-points. Within each bin, questions are averaged within each contributing meeting before meetings are averaged.

### By recall gain

| Recall gain | Questions | Meetings | dRecall | dPrecision | dROUGE-L | dBERTScore F1 | dJudge |
|---|---:|---:|---:|---:|---:|---:|---:|
| small (< 0.10) | 141 | 33 | +0.015 [+0.010, +0.021] | -0.141 [-0.164, -0.120] | -0.002 [-0.012, +0.007] | -0.008 [-0.022, +0.003] | -0.048 [-0.103, +0.009] |
| moderate (0.10 to < 0.25) | 38 | 23 | +0.181 [+0.165, +0.197] | -0.051 [-0.094, -0.002] | +0.003 [-0.011, +0.018] | -0.003 [-0.026, +0.021] | -0.076 [-0.217, +0.036] |
| large (>= 0.25) | 65 | 30 | +0.574 [+0.506, +0.647] | +0.044 [+0.015, +0.072] | +0.016 [+0.000, +0.033] | +0.026 [+0.006, +0.048] | +0.392 [+0.186, +0.614] |

### By precision loss

Precision loss is `precision at 1,024 - precision at 2,048`, so a larger positive value means more added noise under the larger budget. The small stratum also includes questions whose precision increased.

| Precision loss | Questions | Meetings | dRecall | dPrecision | dROUGE-L | dBERTScore F1 | dJudge |
|---|---:|---:|---:|---:|---:|---:|---:|
| small (< 0.05) | 96 | 34 | +0.415 [+0.327, +0.509] | +0.069 [+0.052, +0.086] | +0.011 [-0.000, +0.024] | +0.027 [+0.010, +0.045] | +0.191 [+0.036, +0.367] |
| moderate (0.05 to < 0.15) | 77 | 32 | +0.056 [+0.031, +0.083] | -0.099 [-0.106, -0.092] | -0.001 [-0.015, +0.010] | -0.016 [-0.036, -0.000] | -0.095 [-0.211, +0.007] |
| large (>= 0.15) | 71 | 31 | +0.056 [+0.037, +0.078] | -0.242 [-0.263, -0.223] | -0.003 [-0.016, +0.011] | -0.010 [-0.025, +0.005] | -0.018 [-0.124, +0.077] |

### Question-level rank associations

These are pooled-question Spearman correlations. Their intervals resample whole meetings, keeping all questions from a meeting together.

| Predictor | Outcome | Spearman rho [95% CI] |
|---|---|---:|
| Recall gain | rougeL | +0.067 [-0.082, +0.205] |
| Recall gain | bertscore_f1 | +0.098 [-0.058, +0.250] |
| Recall gain | judge | +0.349 [+0.235, +0.452] |
| Precision loss | rougeL | -0.074 [-0.223, +0.071] |
| Precision loss | bertscore_f1 | -0.236 [-0.383, -0.073] |
| Precision loss | judge | -0.210 [-0.351, -0.056] |

## Reading the result

- In the focused setting, doubling the evidence budget increased recall by +0.192 [+0.161, +0.223], while the judge changed by only +0.039 [-0.021, +0.098]. ROUGE-L changed by +0.002 [-0.004, +0.009] and BERTScore F1 by +0.002 [-0.006, +0.010].
- 65 questions had a recall gain of at least 0.25. Their mean judge change was +0.392 [+0.186, +0.614].
- 71 questions lost at least 0.15 precision. Their mean judge change was -0.018 [-0.124, +0.077].
- Recall gain had a positive rank association with judge change (+0.349, 95% CI [+0.235, +0.452]); precision loss had a negative association (-0.210, 95% CI [-0.351, -0.056]). This is consistent with useful large coverage gains being offset when the additional context is mostly irrelevant.
- A positive retrieval delta with a downstream interval spanning zero supports the interpretation that extra annotated evidence did not translate reliably into better answers under that condition; it does not prove the extra evidence was unused on every question.
- Evidence precision and recall are word-level proxies based on QMSum's annotated turns. Judge scores are ordinal (1-3), although their paired differences are summarized numerically here. The associations are exploratory and not causal.

## Reproduce

```powershell
python src/tools/analyze_budget_deltas.py `
  --preset src/configs/ablation-full.toml `
  --output runs/ablations/full/budget-deltas.json `
  --samples 10000 --seed 42
```

The sibling JSON contains every paired question-level row and all estimates used in this report.
