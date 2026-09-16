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
| lumber | dense | 512 | 0.386 | 0.332 | 0.641 | 0.318 | 0.078 | 0.197 | 0.201 | 1.696 | 60/67/15 |
| lumber | dense | 1024 | 0.326 | 0.521 | 0.641 | 0.335 | 0.088 | 0.205 | 0.214 | 1.909 | 41/71/30 |
| lumber | dense | 2048 | 0.229 | 0.673 | 0.641 | 0.348 | 0.095 | 0.213 | 0.225 | 1.972 | 34/80/28 |
| lumber | bm25 | 512 | 0.329 | 0.293 | 0.611 | 0.313 | 0.081 | 0.197 | 0.192 | 1.693 | 58/65/19 |
| lumber | bm25 | 1024 | 0.261 | 0.471 | 0.611 | 0.328 | 0.084 | 0.203 | 0.210 | 1.886 | 40/77/25 |
| lumber | bm25 | 2048 | 0.207 | 0.620 | 0.611 | 0.345 | 0.094 | 0.214 | 0.225 | 1.981 | 35/77/30 |
| lumber | hybrid | 512 | 0.349 | 0.305 | 0.650 | 0.314 | 0.078 | 0.195 | 0.198 | 1.742 | 58/62/22 |
| lumber | hybrid | 1024 | 0.306 | 0.516 | 0.650 | 0.333 | 0.083 | 0.204 | 0.214 | 1.908 | 37/80/25 |
| lumber | hybrid | 2048 | 0.226 | 0.655 | 0.650 | 0.347 | 0.093 | 0.214 | 0.220 | 1.996 | 34/78/30 |
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
| lumber_minus_single_turn__dense__w512 | 0.090 | 0.088 | 0.001 | 0.001 | -0.112 |
| lumber_minus_turn_packed__dense__w512 | 0.022 | 0.008 | -0.002 | -0.004 | -0.046 |
| lumber_minus_word_packed__dense__w512 | 0.041 | 0.024 | 0.002 | 0.001 | -0.064 |
| lumber_minus_single_turn__dense__w1024 | 0.085 | 0.152 | 0.001 | 0.003 | 0.031 |
| lumber_minus_turn_packed__dense__w1024 | 0.032 | 0.034 | -0.006 | -0.002 | 0.021 |
| lumber_minus_word_packed__dense__w1024 | 0.036 | 0.030 | -0.000 | 0.001 | 0.033 |
| lumber_minus_single_turn__dense__w2048 | 0.039 | 0.132 | 0.006 | 0.014 | -0.000 |
| lumber_minus_turn_packed__dense__w2048 | 0.010 | 0.015 | -0.001 | -0.001 | -0.005 |
| lumber_minus_word_packed__dense__w2048 | 0.003 | -0.003 | 0.003 | 0.006 | 0.042 |
| lumber_minus_single_turn__bm25__w512 | 0.095 | 0.100 | 0.002 | 0.009 | 0.138 |
| lumber_minus_turn_packed__bm25__w512 | 0.042 | 0.024 | 0.001 | 0.002 | 0.031 |
| lumber_minus_word_packed__bm25__w512 | 0.039 | 0.026 | 0.000 | -0.000 | 0.018 |
| lumber_minus_single_turn__bm25__w1024 | 0.063 | 0.165 | 0.003 | 0.011 | 0.147 |
| lumber_minus_turn_packed__bm25__w1024 | 0.025 | 0.045 | -0.004 | 0.002 | 0.073 |
| lumber_minus_word_packed__bm25__w1024 | 0.035 | 0.065 | -0.001 | 0.006 | 0.037 |
| lumber_minus_single_turn__bm25__w2048 | 0.047 | 0.163 | 0.007 | 0.020 | 0.108 |
| lumber_minus_turn_packed__bm25__w2048 | 0.018 | 0.045 | 0.005 | 0.010 | 0.067 |
| lumber_minus_word_packed__bm25__w2048 | 0.015 | 0.047 | 0.001 | 0.006 | 0.024 |
| lumber_minus_single_turn__hybrid__w512 | 0.060 | 0.069 | -0.003 | 0.000 | 0.081 |
| lumber_minus_turn_packed__hybrid__w512 | 0.008 | -0.019 | -0.003 | -0.003 | -0.036 |
| lumber_minus_word_packed__hybrid__w512 | 0.010 | -0.009 | -0.005 | -0.004 | -0.043 |
| lumber_minus_single_turn__hybrid__w1024 | 0.070 | 0.148 | 0.002 | 0.005 | 0.029 |
| lumber_minus_turn_packed__hybrid__w1024 | 0.028 | 0.037 | 0.001 | 0.003 | -0.045 |
| lumber_minus_word_packed__hybrid__w1024 | 0.032 | 0.041 | -0.003 | -0.005 | 0.016 |
| lumber_minus_single_turn__hybrid__w2048 | 0.046 | 0.133 | 0.004 | 0.004 | 0.073 |
| lumber_minus_turn_packed__hybrid__w2048 | 0.011 | 0.020 | 0.007 | 0.005 | -0.018 |
| lumber_minus_word_packed__hybrid__w2048 | 0.010 | 0.003 | 0.005 | -0.002 | 0.053 |

## Best observed configurations

- **Retrieval recall:** `word_packed__dense__w2048` (0.676)
- **ROUGE-L:** `lumber__hybrid__w2048` (0.214)
- **BERTScore F1:** `turn_packed__dense__w2048` (0.227)
- **LLM judge:** `turn_packed__hybrid__w2048` (2.014)

## Interpretation notes

- Retrieval precision and recall are word-weighted against QMSum's annotated evidence spans. First-overlap MRR structurally favours larger chunks and is diagnostic only.
- Retrieval chooses evidence under the budget, then renders selected fragments chronologically for conversational coherence.
- ROUGE and BERTScore compare generated answers with the reference answers. The judge uses the reference answer and gold transcript evidence on a 1--3 scale.
- The 1--3 judge compresses correctness, completeness, and grounding into one ordinal score; manual review remains necessary.
- The judge checkpoint is `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`, separate from the candidate checkpoints.
