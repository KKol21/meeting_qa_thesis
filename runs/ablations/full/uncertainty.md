# Ablation uncertainty analysis

**Dataset split:** QMSum test.

**Scope:** 35 meetings; 10,000 paired cluster-bootstrap samples; seed 42.

Meetings, not questions, are resampled. Thus questions from the same meeting remain together. Point estimates are meeting-macro means; brackets are percentile 95% confidence intervals.

## Headline average Lumber effect

For each meeting and baseline, Lumber-minus-baseline recall is calculated separately in every retriever x evidence-budget environment and then averaged across those environments. These meeting-level average effects are finally averaged and cluster-bootstrapped across meetings.

- Across the 9 prespecified retrieval environments, Lumber increased evidence recall by 1.1 percentage points relative to turn_packed, 95% CI [-0.8, +3.0].
- Across the 9 prespecified retrieval environments, Lumber increased evidence recall by 2.3 percentage points relative to word_packed, 95% CI [-0.1, +4.8].

## All retrieval conditions

| Condition | Recall [95% CI] | MRR [95% CI] | ROUGE-L [95% CI] | BERTScore F1 [95% CI] | Judge [95% CI] |
|---|---:|---:|---:|---:|---:|
| turn_packed__dense__w512 | 0.346 [0.298, 0.397] | 0.681 [0.627, 0.732] | 0.198 [0.187, 0.210] | 0.205 [0.189, 0.221] | 1.809 [1.718, 1.899] |
| turn_packed__dense__w1024 | 0.505 [0.448, 0.563] | 0.681 [0.627, 0.732] | 0.202 [0.191, 0.212] | 0.213 [0.196, 0.230] | 1.902 [1.802, 2.003] |
| turn_packed__dense__w2048 | 0.671 [0.606, 0.733] | 0.681 [0.627, 0.732] | 0.206 [0.197, 0.216] | 0.220 [0.205, 0.236] | 1.956 [1.858, 2.052] |
| turn_packed__bm25__w512 | 0.291 [0.241, 0.342] | 0.628 [0.576, 0.679] | 0.193 [0.184, 0.203] | 0.199 [0.183, 0.215] | 1.742 [1.664, 1.820] |
| turn_packed__bm25__w1024 | 0.449 [0.398, 0.499] | 0.628 [0.576, 0.679] | 0.203 [0.192, 0.213] | 0.215 [0.200, 0.229] | 1.868 [1.779, 1.953] |
| turn_packed__bm25__w2048 | 0.598 [0.534, 0.661] | 0.628 [0.576, 0.679] | 0.207 [0.197, 0.217] | 0.217 [0.201, 0.232] | 1.912 [1.820, 2.002] |
| turn_packed__hybrid__w512 | 0.335 [0.288, 0.384] | 0.687 [0.635, 0.736] | 0.197 [0.187, 0.207] | 0.205 [0.190, 0.220] | 1.790 [1.700, 1.878] |
| turn_packed__hybrid__w1024 | 0.525 [0.458, 0.590] | 0.687 [0.635, 0.736] | 0.201 [0.190, 0.212] | 0.216 [0.199, 0.233] | 1.933 [1.829, 2.030] |
| turn_packed__hybrid__w2048 | 0.665 [0.601, 0.727] | 0.687 [0.635, 0.736] | 0.205 [0.195, 0.216] | 0.220 [0.203, 0.237] | 1.989 [1.889, 2.083] |
| word_packed__dense__w512 | 0.346 [0.297, 0.397] | 0.699 [0.634, 0.759] | 0.199 [0.188, 0.210] | 0.210 [0.196, 0.225] | 1.820 [1.721, 1.920] |
| word_packed__dense__w1024 | 0.486 [0.428, 0.545] | 0.699 [0.634, 0.759] | 0.203 [0.193, 0.213] | 0.216 [0.198, 0.233] | 1.956 [1.852, 2.055] |
| word_packed__dense__w2048 | 0.672 [0.609, 0.732] | 0.699 [0.634, 0.759] | 0.208 [0.198, 0.218] | 0.223 [0.206, 0.240] | 1.974 [1.874, 2.073] |
| word_packed__bm25__w512 | 0.295 [0.249, 0.341] | 0.665 [0.611, 0.718] | 0.194 [0.185, 0.204] | 0.203 [0.186, 0.219] | 1.804 [1.716, 1.891] |
| word_packed__bm25__w1024 | 0.435 [0.376, 0.498] | 0.665 [0.611, 0.718] | 0.199 [0.189, 0.209] | 0.211 [0.197, 0.226] | 1.866 [1.768, 1.966] |
| word_packed__bm25__w2048 | 0.586 [0.520, 0.652] | 0.665 [0.611, 0.718] | 0.202 [0.191, 0.212] | 0.216 [0.200, 0.232] | 1.932 [1.848, 2.012] |
| word_packed__hybrid__w512 | 0.341 [0.293, 0.392] | 0.731 [0.669, 0.789] | 0.198 [0.187, 0.209] | 0.209 [0.194, 0.225] | 1.868 [1.777, 1.960] |
| word_packed__hybrid__w1024 | 0.468 [0.407, 0.531] | 0.731 [0.669, 0.789] | 0.202 [0.191, 0.212] | 0.219 [0.205, 0.233] | 1.925 [1.819, 2.029] |
| word_packed__hybrid__w2048 | 0.648 [0.581, 0.711] | 0.731 [0.669, 0.789] | 0.206 [0.196, 0.217] | 0.219 [0.204, 0.234] | 1.958 [1.848, 2.057] |
| lumber__dense__w512 | 0.350 [0.293, 0.410] | 0.682 [0.626, 0.737] | 0.198 [0.190, 0.206] | 0.209 [0.195, 0.223] | 1.796 [1.700, 1.893] |
| lumber__dense__w1024 | 0.497 [0.436, 0.559] | 0.682 [0.626, 0.737] | 0.201 [0.191, 0.211] | 0.216 [0.202, 0.232] | 1.952 [1.864, 2.042] |
| lumber__dense__w2048 | 0.689 [0.632, 0.745] | 0.682 [0.626, 0.737] | 0.204 [0.194, 0.214] | 0.219 [0.204, 0.233] | 1.991 [1.892, 2.088] |
| lumber__bm25__w512 | 0.326 [0.273, 0.381] | 0.649 [0.593, 0.704] | 0.193 [0.184, 0.202] | 0.200 [0.183, 0.217] | 1.767 [1.672, 1.865] |
| lumber__bm25__w1024 | 0.467 [0.406, 0.526] | 0.649 [0.593, 0.704] | 0.197 [0.187, 0.206] | 0.212 [0.197, 0.227] | 1.851 [1.754, 1.943] |
| lumber__bm25__w2048 | 0.633 [0.564, 0.703] | 0.649 [0.593, 0.704] | 0.206 [0.196, 0.217] | 0.222 [0.206, 0.237] | 1.952 [1.857, 2.042] |
| lumber__hybrid__w512 | 0.355 [0.298, 0.414] | 0.711 [0.655, 0.762] | 0.191 [0.182, 0.200] | 0.205 [0.191, 0.218] | 1.840 [1.747, 1.931] |
| lumber__hybrid__w1024 | 0.510 [0.439, 0.581] | 0.711 [0.655, 0.762] | 0.199 [0.190, 0.210] | 0.217 [0.202, 0.233] | 1.923 [1.835, 2.010] |
| lumber__hybrid__w2048 | 0.659 [0.591, 0.725] | 0.711 [0.655, 0.762] | 0.206 [0.196, 0.215] | 0.220 [0.204, 0.235] | 2.005 [1.915, 2.090] |

## Paired Lumber contrasts

Each value is Lumber minus the named deterministic chunker under the same retriever and evidence budget.

| Comparison | dRecall [95% CI] | dROUGE-L [95% CI] | dBERTScore F1 [95% CI] | dJudge [95% CI] |
|---|---:|---:|---:|---:|
| lumber__dense__w512 minus turn_packed | +0.004 [-0.031, +0.043] | -0.000 [-0.008, +0.007] | +0.004 [-0.009, +0.019] | -0.013 [-0.093, +0.073] |
| lumber__dense__w512 minus word_packed | +0.004 [-0.033, +0.040] | -0.000 [-0.009, +0.008] | -0.001 [-0.014, +0.013] | -0.024 [-0.118, +0.069] |
| lumber__dense__w1024 minus turn_packed | -0.008 [-0.039, +0.022] | -0.001 [-0.007, +0.006] | +0.003 [-0.007, +0.013] | +0.051 [-0.012, +0.115] |
| lumber__dense__w1024 minus word_packed | +0.011 [-0.029, +0.052] | -0.002 [-0.008, +0.004] | +0.001 [-0.011, +0.013] | -0.004 [-0.073, +0.068] |
| lumber__dense__w2048 minus turn_packed | +0.018 [-0.012, +0.046] | -0.003 [-0.008, +0.002] | -0.001 [-0.009, +0.006] | +0.035 [-0.029, +0.096] |
| lumber__dense__w2048 minus word_packed | +0.017 [-0.023, +0.056] | -0.005 [-0.011, +0.002] | -0.004 [-0.015, +0.006] | +0.017 [-0.044, +0.076] |
| lumber__bm25__w512 minus turn_packed | +0.036 [+0.003, +0.067] | -0.001 [-0.008, +0.007] | +0.001 [-0.009, +0.012] | +0.026 [-0.040, +0.095] |
| lumber__bm25__w512 minus word_packed | +0.032 [-0.001, +0.067] | -0.002 [-0.008, +0.004] | -0.002 [-0.012, +0.007] | -0.037 [-0.121, +0.046] |
| lumber__bm25__w1024 minus turn_packed | +0.018 [-0.012, +0.048] | -0.006 [-0.012, -0.000] | -0.003 [-0.012, +0.005] | -0.017 [-0.103, +0.070] |
| lumber__bm25__w1024 minus word_packed | +0.032 [-0.011, +0.074] | -0.002 [-0.008, +0.005] | +0.001 [-0.009, +0.012] | -0.016 [-0.100, +0.076] |
| lumber__bm25__w2048 minus turn_packed | +0.034 [+0.004, +0.064] | -0.000 [-0.006, +0.005] | +0.005 [-0.004, +0.014] | +0.039 [-0.027, +0.110] |
| lumber__bm25__w2048 minus word_packed | +0.047 [+0.014, +0.079] | +0.005 [-0.001, +0.010] | +0.006 [-0.002, +0.013] | +0.019 [-0.034, +0.074] |
| lumber__hybrid__w512 minus turn_packed | +0.020 [-0.013, +0.051] | -0.006 [-0.013, +0.002] | -0.000 [-0.011, +0.010] | +0.050 [-0.026, +0.124] |
| lumber__hybrid__w512 minus word_packed | +0.014 [-0.019, +0.048] | -0.007 [-0.014, +0.001] | -0.005 [-0.015, +0.006] | -0.028 [-0.099, +0.040] |
| lumber__hybrid__w1024 minus turn_packed | -0.015 [-0.045, +0.016] | -0.002 [-0.009, +0.005] | +0.002 [-0.010, +0.013] | -0.011 [-0.074, +0.055] |
| lumber__hybrid__w1024 minus word_packed | +0.042 [+0.002, +0.081] | -0.002 [-0.008, +0.004] | -0.002 [-0.011, +0.008] | -0.003 [-0.074, +0.071] |
| lumber__hybrid__w2048 minus turn_packed | -0.006 [-0.031, +0.018] | +0.000 [-0.005, +0.005] | -0.000 [-0.011, +0.010] | +0.016 [-0.052, +0.082] |
| lumber__hybrid__w2048 minus word_packed | +0.011 [-0.023, +0.046] | -0.001 [-0.007, +0.005] | +0.001 [-0.009, +0.010] | +0.047 [-0.045, +0.128] |

## Oracle answer models

| Stage | ROUGE-L [95% CI] | BERTScore F1 [95% CI] | Judge [95% CI] |
|---|---:|---:|---:|
| oracle-7b | 0.231 [0.222, 0.241] | 0.249 [0.234, 0.265] | 2.512 [2.427, 2.597] |
| oracle-14b | 0.229 [0.219, 0.240] | 0.262 [0.246, 0.277] | 2.493 [2.403, 2.581] |
| oracle-32b-bnb4 | 0.224 [0.213, 0.234] | 0.239 [0.221, 0.256] | 2.594 [2.482, 2.696] |

### Paired oracle-model differences

| Comparison | dROUGE-L [95% CI] | dBERTScore F1 [95% CI] | dJudge [95% CI] |
|---|---:|---:|---:|
| oracle-14b minus oracle-7b | -0.002 [-0.010, +0.007] | +0.013 [+0.001, +0.024] | -0.018 [-0.100, +0.066] |
| oracle-32b-bnb4 minus oracle-7b | -0.007 [-0.016, +0.001] | -0.011 [-0.024, +0.003] | +0.083 [-0.014, +0.180] |
| oracle-32b-bnb4 minus oracle-14b | -0.005 [-0.014, +0.003] | -0.023 [-0.034, -0.013] | +0.101 [+0.022, +0.179] |

## Oracle evidence gap

Paired difference: **oracle-14b minus retrieval-14b / lumber__dense__w1024**. Both stages use the same answer model, so this isolates the downstream association with evidence source.

| dROUGE-L [95% CI] | dBERTScore F1 [95% CI] | dJudge [95% CI] |
|---:|---:|---:|
| +0.028 [+0.019, +0.038] | +0.045 [+0.030, +0.060] | +0.541 [+0.436, +0.647] |

## Interpretation limits

An interval excluding zero is evidence that the paired meeting-level difference is consistently directional under this dataset sample. It is not a correction for the many exploratory comparisons, and only 35 meeting clusters are available. Judge and BERTScore uncertainty also does not include uncertainty from changing the evaluator model or prompt.
