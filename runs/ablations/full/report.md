# Ablation report

**Dataset split:** QMSum test.

**Scope:** 35 meeting(s), 244 question(s).

## Oracle answer-model comparison

Gold evidence is supplied here, isolating answer-model performance from retrieval.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Judge mean | Judge 1/2/3 |
|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-14b | 0.359 | 0.114 | 0.229 | 0.262 | 2.493 | 13/88/143 |
| qwen2.5-32b-bnb4 | 0.349 | 0.115 | 0.224 | 0.239 | 2.594 | 18/56/170 |
| qwen2.5-7b | 0.356 | 0.118 | 0.231 | 0.249 | 2.512 | 9/96/139 |

## Retrieval and end-to-end comparison

Means are meeting-macro averages; judge 1/2/3 counts are question totals. First-overlap MRR is retained only as a size-sensitive diagnostic.

| Chunker | Retriever | Words | Precision | Recall | First-overlap MRR | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Judge | 1/2/3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| turn_packed | dense | 512 | 0.357 | 0.346 | 0.681 | 0.312 | 0.083 | 0.198 | 0.205 | 1.809 | 72/143/29 |
| turn_packed | dense | 1024 | 0.280 | 0.505 | 0.681 | 0.321 | 0.090 | 0.202 | 0.213 | 1.902 | 59/143/42 |
| turn_packed | dense | 2048 | 0.212 | 0.671 | 0.681 | 0.327 | 0.092 | 0.206 | 0.220 | 1.956 | 50/148/46 |
| turn_packed | bm25 | 512 | 0.303 | 0.291 | 0.628 | 0.309 | 0.080 | 0.193 | 0.199 | 1.742 | 83/135/26 |
| turn_packed | bm25 | 1024 | 0.250 | 0.449 | 0.628 | 0.319 | 0.086 | 0.203 | 0.215 | 1.868 | 61/147/36 |
| turn_packed | bm25 | 2048 | 0.190 | 0.598 | 0.628 | 0.327 | 0.093 | 0.207 | 0.217 | 1.912 | 56/145/43 |
| turn_packed | hybrid | 512 | 0.346 | 0.335 | 0.687 | 0.311 | 0.083 | 0.197 | 0.205 | 1.790 | 69/148/27 |
| turn_packed | hybrid | 1024 | 0.291 | 0.525 | 0.687 | 0.324 | 0.089 | 0.201 | 0.216 | 1.933 | 50/151/43 |
| turn_packed | hybrid | 2048 | 0.212 | 0.665 | 0.687 | 0.328 | 0.095 | 0.205 | 0.220 | 1.989 | 42/155/47 |
| word_packed | dense | 512 | 0.360 | 0.346 | 0.699 | 0.315 | 0.085 | 0.199 | 0.210 | 1.820 | 76/131/37 |
| word_packed | dense | 1024 | 0.285 | 0.486 | 0.699 | 0.323 | 0.090 | 0.203 | 0.216 | 1.956 | 48/153/43 |
| word_packed | dense | 2048 | 0.220 | 0.672 | 0.699 | 0.331 | 0.094 | 0.208 | 0.223 | 1.974 | 49/146/49 |
| word_packed | bm25 | 512 | 0.314 | 0.295 | 0.665 | 0.308 | 0.081 | 0.194 | 0.203 | 1.804 | 77/131/36 |
| word_packed | bm25 | 1024 | 0.253 | 0.435 | 0.665 | 0.317 | 0.085 | 0.199 | 0.211 | 1.866 | 59/147/38 |
| word_packed | bm25 | 2048 | 0.187 | 0.586 | 0.665 | 0.323 | 0.087 | 0.202 | 0.216 | 1.932 | 53/150/41 |
| word_packed | hybrid | 512 | 0.349 | 0.341 | 0.731 | 0.313 | 0.084 | 0.198 | 0.209 | 1.868 | 65/137/42 |
| word_packed | hybrid | 1024 | 0.268 | 0.468 | 0.731 | 0.320 | 0.086 | 0.202 | 0.219 | 1.925 | 57/140/47 |
| word_packed | hybrid | 2048 | 0.210 | 0.648 | 0.731 | 0.328 | 0.092 | 0.206 | 0.219 | 1.958 | 51/146/47 |
| lumber | dense | 512 | 0.362 | 0.350 | 0.682 | 0.314 | 0.082 | 0.198 | 0.209 | 1.796 | 83/122/39 |
| lumber | dense | 1024 | 0.294 | 0.497 | 0.682 | 0.323 | 0.088 | 0.201 | 0.216 | 1.952 | 56/142/46 |
| lumber | dense | 2048 | 0.223 | 0.689 | 0.682 | 0.325 | 0.090 | 0.204 | 0.219 | 1.991 | 53/135/56 |
| lumber | bm25 | 512 | 0.341 | 0.326 | 0.649 | 0.309 | 0.079 | 0.193 | 0.200 | 1.767 | 89/113/42 |
| lumber | bm25 | 1024 | 0.264 | 0.467 | 0.649 | 0.319 | 0.086 | 0.197 | 0.212 | 1.851 | 67/136/41 |
| lumber | bm25 | 2048 | 0.200 | 0.633 | 0.649 | 0.325 | 0.091 | 0.206 | 0.222 | 1.952 | 56/135/53 |
| lumber | hybrid | 512 | 0.381 | 0.355 | 0.711 | 0.308 | 0.078 | 0.191 | 0.205 | 1.840 | 74/127/43 |
| lumber | hybrid | 1024 | 0.289 | 0.510 | 0.711 | 0.323 | 0.089 | 0.199 | 0.217 | 1.923 | 55/146/43 |
| lumber | hybrid | 2048 | 0.212 | 0.659 | 0.711 | 0.327 | 0.092 | 0.206 | 0.220 | 2.005 | 43/148/53 |

## Paired meeting-level Lumber differences

Positive values favour Lumber. Each value is the mean of within-meeting differences.

| Comparison | Precision | Recall | ROUGE-L | BERTScore F1 | Judge |
|---|---:|---:|---:|---:|---:|
| lumber_minus_turn_packed__dense__w512 | 0.005 | 0.004 | -0.000 | 0.004 | -0.013 |
| lumber_minus_word_packed__dense__w512 | 0.002 | 0.004 | -0.000 | -0.001 | -0.024 |
| lumber_minus_turn_packed__dense__w1024 | 0.014 | -0.008 | -0.001 | 0.003 | 0.051 |
| lumber_minus_word_packed__dense__w1024 | 0.009 | 0.011 | -0.002 | 0.001 | -0.004 |
| lumber_minus_turn_packed__dense__w2048 | 0.011 | 0.018 | -0.003 | -0.001 | 0.035 |
| lumber_minus_word_packed__dense__w2048 | 0.003 | 0.017 | -0.005 | -0.004 | 0.017 |
| lumber_minus_turn_packed__bm25__w512 | 0.038 | 0.036 | -0.001 | 0.001 | 0.026 |
| lumber_minus_word_packed__bm25__w512 | 0.026 | 0.032 | -0.002 | -0.002 | -0.037 |
| lumber_minus_turn_packed__bm25__w1024 | 0.014 | 0.018 | -0.006 | -0.003 | -0.017 |
| lumber_minus_word_packed__bm25__w1024 | 0.011 | 0.032 | -0.002 | 0.001 | -0.016 |
| lumber_minus_turn_packed__bm25__w2048 | 0.009 | 0.034 | -0.000 | 0.005 | 0.039 |
| lumber_minus_word_packed__bm25__w2048 | 0.013 | 0.047 | 0.005 | 0.006 | 0.019 |
| lumber_minus_turn_packed__hybrid__w512 | 0.034 | 0.020 | -0.006 | -0.000 | 0.050 |
| lumber_minus_word_packed__hybrid__w512 | 0.032 | 0.014 | -0.007 | -0.005 | -0.028 |
| lumber_minus_turn_packed__hybrid__w1024 | -0.002 | -0.015 | -0.002 | 0.002 | -0.011 |
| lumber_minus_word_packed__hybrid__w1024 | 0.021 | 0.042 | -0.002 | -0.002 | -0.003 |
| lumber_minus_turn_packed__hybrid__w2048 | 0.001 | -0.006 | 0.000 | -0.000 | 0.016 |
| lumber_minus_word_packed__hybrid__w2048 | 0.002 | 0.011 | -0.001 | 0.001 | 0.047 |

## Best observed configurations

- **Retrieval recall:** `lumber__dense__w2048` (0.689)
- **ROUGE-L:** `word_packed__dense__w2048` (0.208)
- **BERTScore F1:** `word_packed__dense__w2048` (0.223)
- **LLM judge:** `lumber__hybrid__w2048` (2.005)

## Interpretation notes

- Retrieval precision and recall are word-weighted against QMSum's annotated evidence spans. First-overlap MRR structurally favours larger chunks and is diagnostic only.
- Retrieval chooses evidence under the budget, then renders selected fragments chronologically for conversational coherence.
- ROUGE and BERTScore compare generated answers with the reference answers. The judge uses the reference answer and gold transcript evidence on a 1--3 scale.
- The 1--3 judge compresses correctness, completeness, and grounding into one ordinal score; manual review remains necessary.
- The judge checkpoint is `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`, separate from the candidate checkpoints.
