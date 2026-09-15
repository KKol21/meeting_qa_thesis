# Ablation report

**Scope:** 20 meeting(s), 142 question(s).

## Oracle answer-model comparison

Gold evidence is supplied here, isolating answer-model performance from retrieval.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Judge mean | Judge 1/2/3 |
|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-14b | 0.360 | 0.104 | 0.228 | 0.259 | 2.533 | 5/57/80 |
| qwen2.5-32b-bnb4 | 0.362 | 0.109 | 0.225 | 0.239 | 2.586 | 12/35/95 |
| qwen2.5-7b | 0.367 | 0.107 | 0.227 | 0.240 | 2.458 | 12/56/74 |

## Retrieval and end-to-end comparison

Means are meeting-macro averages; judge 1/2/3 counts are question totals. First-overlap MRR is retained only as a size-sensitive diagnostic.

| Chunker | Retriever | Words | Precision | Recall | First-overlap MRR | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Judge | 1/2/3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| turn_packed | dense | 512 | 0.363 | 0.324 | 0.685 | 0.318 | 0.083 | 0.199 | 0.204 | 1.742 | 48/82/12 |
| turn_packed | dense | 1024 | 0.294 | 0.487 | 0.685 | 0.340 | 0.093 | 0.211 | 0.217 | 1.888 | 35/87/20 |
| turn_packed | dense | 2048 | 0.219 | 0.658 | 0.685 | 0.350 | 0.096 | 0.214 | 0.227 | 1.977 | 34/79/29 |
| turn_packed | bm25 | 512 | 0.287 | 0.269 | 0.602 | 0.309 | 0.080 | 0.197 | 0.190 | 1.662 | 56/75/11 |
| turn_packed | bm25 | 1024 | 0.236 | 0.426 | 0.602 | 0.332 | 0.088 | 0.207 | 0.208 | 1.813 | 41/86/15 |
| turn_packed | bm25 | 2048 | 0.189 | 0.576 | 0.602 | 0.335 | 0.091 | 0.209 | 0.215 | 1.914 | 35/87/20 |
| turn_packed | hybrid | 512 | 0.341 | 0.324 | 0.644 | 0.315 | 0.083 | 0.198 | 0.201 | 1.778 | 45/84/13 |
| turn_packed | hybrid | 1024 | 0.277 | 0.479 | 0.644 | 0.333 | 0.085 | 0.203 | 0.211 | 1.953 | 33/82/27 |
| turn_packed | hybrid | 2048 | 0.215 | 0.635 | 0.644 | 0.341 | 0.090 | 0.207 | 0.215 | 2.014 | 30/83/29 |
| word_packed | dense | 512 | 0.344 | 0.308 | 0.696 | 0.315 | 0.077 | 0.195 | 0.200 | 1.761 | 46/83/13 |
| word_packed | dense | 1024 | 0.290 | 0.491 | 0.696 | 0.329 | 0.082 | 0.205 | 0.213 | 1.876 | 42/77/23 |
| word_packed | dense | 2048 | 0.226 | 0.676 | 0.696 | 0.342 | 0.092 | 0.210 | 0.219 | 1.930 | 39/74/29 |
| word_packed | bm25 | 512 | 0.290 | 0.267 | 0.620 | 0.315 | 0.081 | 0.197 | 0.192 | 1.675 | 54/79/9 |
| word_packed | bm25 | 1024 | 0.226 | 0.406 | 0.620 | 0.323 | 0.082 | 0.204 | 0.205 | 1.849 | 42/79/21 |
| word_packed | bm25 | 2048 | 0.191 | 0.573 | 0.620 | 0.340 | 0.094 | 0.213 | 0.219 | 1.958 | 32/85/25 |
| word_packed | hybrid | 512 | 0.339 | 0.315 | 0.712 | 0.322 | 0.080 | 0.200 | 0.202 | 1.784 | 45/83/14 |
| word_packed | hybrid | 1024 | 0.274 | 0.475 | 0.712 | 0.339 | 0.089 | 0.207 | 0.219 | 1.892 | 37/83/22 |
| word_packed | hybrid | 2048 | 0.216 | 0.653 | 0.712 | 0.342 | 0.092 | 0.209 | 0.222 | 1.943 | 36/78/28 |
| lumber | dense | 512 | 0.368 | 0.350 | 0.696 | 0.324 | 0.084 | 0.201 | 0.210 | 1.740 | 46/85/11 |
| lumber | dense | 1024 | 0.308 | 0.535 | 0.696 | 0.342 | 0.092 | 0.213 | 0.228 | 2.032 | 24/90/28 |
| lumber | dense | 2048 | 0.228 | 0.686 | 0.696 | 0.348 | 0.093 | 0.212 | 0.226 | 2.007 | 25/93/24 |
| lumber | bm25 | 512 | 0.288 | 0.278 | 0.578 | 0.317 | 0.081 | 0.201 | 0.201 | 1.701 | 52/75/15 |
| lumber | bm25 | 1024 | 0.243 | 0.422 | 0.578 | 0.330 | 0.085 | 0.206 | 0.213 | 1.890 | 42/75/25 |
| lumber | bm25 | 2048 | 0.188 | 0.581 | 0.578 | 0.339 | 0.092 | 0.210 | 0.218 | 1.957 | 33/87/22 |
| lumber | hybrid | 512 | 0.350 | 0.337 | 0.708 | 0.330 | 0.084 | 0.205 | 0.212 | 1.791 | 42/84/16 |
| lumber | hybrid | 1024 | 0.281 | 0.479 | 0.708 | 0.337 | 0.088 | 0.204 | 0.221 | 1.917 | 35/80/27 |
| lumber | hybrid | 2048 | 0.221 | 0.662 | 0.708 | 0.353 | 0.095 | 0.216 | 0.226 | 2.002 | 32/79/31 |
| single_turn | dense | 512 | 0.295 | 0.243 | 0.535 | 0.321 | 0.079 | 0.196 | 0.199 | 1.808 | 41/86/15 |
| single_turn | dense | 1024 | 0.241 | 0.369 | 0.535 | 0.331 | 0.085 | 0.204 | 0.212 | 1.878 | 34/89/19 |
| single_turn | dense | 2048 | 0.190 | 0.540 | 0.535 | 0.339 | 0.089 | 0.207 | 0.212 | 1.973 | 30/88/24 |
| single_turn | bm25 | 512 | 0.233 | 0.192 | 0.490 | 0.307 | 0.075 | 0.195 | 0.183 | 1.555 | 62/76/4 |
| single_turn | bm25 | 1024 | 0.198 | 0.306 | 0.490 | 0.319 | 0.084 | 0.199 | 0.200 | 1.739 | 48/82/12 |
| single_turn | bm25 | 2048 | 0.159 | 0.457 | 0.490 | 0.334 | 0.088 | 0.207 | 0.205 | 1.873 | 38/87/17 |
| single_turn | hybrid | 512 | 0.289 | 0.236 | 0.568 | 0.317 | 0.079 | 0.198 | 0.197 | 1.661 | 50/87/5 |
| single_turn | hybrid | 1024 | 0.235 | 0.368 | 0.568 | 0.329 | 0.085 | 0.202 | 0.209 | 1.879 | 37/85/20 |
| single_turn | hybrid | 2048 | 0.180 | 0.522 | 0.568 | 0.344 | 0.090 | 0.210 | 0.216 | 1.923 | 39/76/27 |

## Paired meeting-level Lumber differences

Positive values favour Lumber. Each value is the mean of within-meeting differences.

| Comparison | Precision | Recall | ROUGE-L | BERTScore F1 | Judge |
|---|---:|---:|---:|---:|---:|
| lumber_minus_single_turn__dense__w512 | 0.073 | 0.106 | 0.005 | 0.010 | -0.068 |
| lumber_minus_turn_packed__dense__w512 | 0.005 | 0.025 | 0.002 | 0.005 | -0.002 |
| lumber_minus_word_packed__dense__w512 | 0.024 | 0.042 | 0.006 | 0.010 | -0.021 |
| lumber_minus_single_turn__dense__w1024 | 0.067 | 0.166 | 0.010 | 0.017 | 0.154 |
| lumber_minus_turn_packed__dense__w1024 | 0.014 | 0.048 | 0.002 | 0.011 | 0.143 |
| lumber_minus_word_packed__dense__w1024 | 0.017 | 0.044 | 0.008 | 0.015 | 0.156 |
| lumber_minus_single_turn__dense__w2048 | 0.039 | 0.145 | 0.005 | 0.014 | 0.035 |
| lumber_minus_turn_packed__dense__w2048 | 0.009 | 0.028 | -0.002 | -0.000 | 0.030 |
| lumber_minus_word_packed__dense__w2048 | 0.003 | 0.010 | 0.002 | 0.007 | 0.077 |
| lumber_minus_single_turn__bm25__w512 | 0.055 | 0.086 | 0.006 | 0.018 | 0.146 |
| lumber_minus_turn_packed__bm25__w512 | 0.001 | 0.009 | 0.005 | 0.011 | 0.039 |
| lumber_minus_word_packed__bm25__w512 | -0.002 | 0.012 | 0.004 | 0.008 | 0.026 |
| lumber_minus_single_turn__bm25__w1024 | 0.045 | 0.115 | 0.007 | 0.013 | 0.151 |
| lumber_minus_turn_packed__bm25__w1024 | 0.007 | -0.004 | -0.001 | 0.004 | 0.077 |
| lumber_minus_word_packed__bm25__w1024 | 0.016 | 0.016 | 0.002 | 0.008 | 0.041 |
| lumber_minus_single_turn__bm25__w2048 | 0.028 | 0.123 | 0.003 | 0.013 | 0.084 |
| lumber_minus_turn_packed__bm25__w2048 | -0.002 | 0.005 | 0.001 | 0.003 | 0.042 |
| lumber_minus_word_packed__bm25__w2048 | -0.004 | 0.007 | -0.003 | -0.001 | -0.001 |
| lumber_minus_single_turn__hybrid__w512 | 0.061 | 0.101 | 0.007 | 0.015 | 0.130 |
| lumber_minus_turn_packed__hybrid__w512 | 0.008 | 0.013 | 0.007 | 0.011 | 0.014 |
| lumber_minus_word_packed__hybrid__w512 | 0.011 | 0.022 | 0.005 | 0.011 | 0.007 |
| lumber_minus_single_turn__hybrid__w1024 | 0.046 | 0.110 | 0.002 | 0.012 | 0.039 |
| lumber_minus_turn_packed__hybrid__w1024 | 0.004 | -0.000 | 0.001 | 0.010 | -0.036 |
| lumber_minus_word_packed__hybrid__w1024 | 0.007 | 0.003 | -0.003 | 0.002 | 0.025 |
| lumber_minus_single_turn__hybrid__w2048 | 0.041 | 0.140 | 0.006 | 0.010 | 0.080 |
| lumber_minus_turn_packed__hybrid__w2048 | 0.006 | 0.027 | 0.009 | 0.011 | -0.011 |
| lumber_minus_word_packed__hybrid__w2048 | 0.005 | 0.010 | 0.007 | 0.005 | 0.060 |

## Best observed configurations

- **Retrieval recall:** `lumber__dense__w2048` (0.686)
- **ROUGE-L:** `lumber__hybrid__w2048` (0.216)
- **BERTScore F1:** `lumber__dense__w1024` (0.228)
- **LLM judge:** `lumber__dense__w1024` (2.032)

## Interpretation notes

- Retrieval precision and recall are word-weighted against QMSum's annotated evidence spans. First-overlap MRR structurally favours larger chunks and is diagnostic only.
- Retrieval chooses evidence under the budget, then renders selected fragments chronologically for conversational coherence.
- ROUGE and BERTScore compare generated answers with the reference answers. The judge uses the reference answer and gold transcript evidence on a 1--3 scale.
- The 1--3 judge compresses correctness, completeness, and grounding into one ordinal score; manual review remains necessary.
- The judge checkpoint is `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`, separate from the candidate checkpoints.
