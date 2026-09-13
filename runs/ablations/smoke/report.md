# Ablation report

**Scope:** 1 meeting(s), 6 question(s).

> This is a smoke test. Treat rankings as pipeline validation, not experimental conclusions.

## Oracle answer-model comparison

Gold evidence is supplied here, isolating answer-model performance from retrieval.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Judge mean | Judge 1/2/3 |
|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-14b | 0.371 | 0.094 | 0.235 | 0.223 | 2.333 | 1/2/3 |
| qwen2.5-32b-bnb4 | 0.355 | 0.072 | 0.213 | 0.192 | 2.167 | 1/3/2 |
| qwen2.5-7b | 0.243 | 0.037 | 0.170 | 0.099 | 1.500 | 4/1/1 |

## Retrieval and end-to-end comparison

Means are meeting-macro averages; judge 1/2/3 counts are question totals. First-overlap MRR is retained only as a size-sensitive diagnostic.

| Chunker | Retriever | Words | Precision | Recall | First-overlap MRR | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Judge | 1/2/3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| turn_packed | dense | 512 | 0.396 | 0.170 | 0.542 | 0.313 | 0.056 | 0.178 | 0.135 | 1.667 | 3/2/1 |
| turn_packed | dense | 1024 | 0.375 | 0.301 | 0.542 | 0.300 | 0.055 | 0.187 | 0.163 | 1.833 | 1/5/0 |
| turn_packed | dense | 2048 | 0.378 | 0.606 | 0.542 | 0.297 | 0.062 | 0.188 | 0.171 | 2.000 | 2/2/2 |
| turn_packed | bm25 | 512 | 0.373 | 0.145 | 0.485 | 0.263 | 0.044 | 0.163 | 0.164 | 1.667 | 4/0/2 |
| turn_packed | bm25 | 1024 | 0.437 | 0.362 | 0.485 | 0.324 | 0.053 | 0.188 | 0.172 | 2.000 | 1/4/1 |
| turn_packed | bm25 | 2048 | 0.267 | 0.413 | 0.485 | 0.306 | 0.061 | 0.178 | 0.181 | 2.000 | 1/4/1 |
| turn_packed | hybrid | 512 | 0.425 | 0.141 | 0.640 | 0.285 | 0.048 | 0.188 | 0.150 | 1.667 | 2/4/0 |
| turn_packed | hybrid | 1024 | 0.397 | 0.264 | 0.640 | 0.302 | 0.050 | 0.185 | 0.183 | 2.000 | 2/2/2 |
| turn_packed | hybrid | 2048 | 0.341 | 0.561 | 0.640 | 0.312 | 0.067 | 0.199 | 0.175 | 2.333 | 1/2/3 |
| word_packed | dense | 512 | 0.477 | 0.207 | 0.708 | 0.253 | 0.034 | 0.166 | 0.121 | 1.333 | 4/2/0 |
| word_packed | dense | 1024 | 0.322 | 0.260 | 0.708 | 0.257 | 0.034 | 0.166 | 0.098 | 1.167 | 5/1/0 |
| word_packed | dense | 2048 | 0.245 | 0.466 | 0.708 | 0.270 | 0.038 | 0.173 | 0.121 | 1.333 | 4/2/0 |
| word_packed | bm25 | 512 | 0.417 | 0.178 | 0.639 | 0.290 | 0.051 | 0.181 | 0.181 | 1.833 | 2/3/1 |
| word_packed | bm25 | 1024 | 0.325 | 0.267 | 0.639 | 0.325 | 0.064 | 0.194 | 0.195 | 2.000 | 1/4/1 |
| word_packed | bm25 | 2048 | 0.207 | 0.321 | 0.639 | 0.327 | 0.079 | 0.189 | 0.202 | 2.000 | 1/4/1 |
| word_packed | hybrid | 512 | 0.250 | 0.109 | 0.575 | 0.249 | 0.039 | 0.170 | 0.139 | 1.500 | 3/3/0 |
| word_packed | hybrid | 1024 | 0.250 | 0.203 | 0.575 | 0.278 | 0.050 | 0.170 | 0.150 | 1.667 | 2/4/0 |
| word_packed | hybrid | 2048 | 0.325 | 0.565 | 0.575 | 0.255 | 0.044 | 0.160 | 0.110 | 1.833 | 2/3/1 |
| lumber | dense | 512 | 0.333 | 0.092 | 0.479 | 0.272 | 0.042 | 0.172 | 0.110 | 1.667 | 3/2/1 |
| lumber | dense | 1024 | 0.342 | 0.211 | 0.479 | 0.271 | 0.045 | 0.176 | 0.122 | 1.667 | 3/2/1 |
| lumber | dense | 2048 | 0.370 | 0.632 | 0.479 | 0.306 | 0.054 | 0.180 | 0.159 | 2.167 | 2/1/3 |
| lumber | bm25 | 512 | 0.285 | 0.114 | 0.456 | 0.261 | 0.054 | 0.163 | 0.192 | 1.500 | 4/1/1 |
| lumber | bm25 | 1024 | 0.414 | 0.365 | 0.456 | 0.308 | 0.057 | 0.187 | 0.190 | 2.167 | 2/1/3 |
| lumber | bm25 | 2048 | 0.294 | 0.497 | 0.456 | 0.318 | 0.055 | 0.186 | 0.193 | 2.333 | 1/2/3 |
| lumber | hybrid | 512 | 0.482 | 0.179 | 0.569 | 0.271 | 0.046 | 0.180 | 0.156 | 1.500 | 4/1/1 |
| lumber | hybrid | 1024 | 0.521 | 0.435 | 0.569 | 0.323 | 0.059 | 0.183 | 0.188 | 2.000 | 1/4/1 |
| lumber | hybrid | 2048 | 0.348 | 0.545 | 0.569 | 0.318 | 0.063 | 0.206 | 0.188 | 2.167 | 1/3/2 |
| single_turn | dense | 512 | 0.458 | 0.144 | 0.471 | 0.277 | 0.051 | 0.175 | 0.169 | 1.833 | 2/3/1 |
| single_turn | dense | 1024 | 0.456 | 0.311 | 0.471 | 0.279 | 0.036 | 0.164 | 0.146 | 1.667 | 3/2/1 |
| single_turn | dense | 2048 | 0.294 | 0.449 | 0.471 | 0.275 | 0.043 | 0.166 | 0.115 | 2.000 | 2/2/2 |
| single_turn | bm25 | 512 | 0.224 | 0.087 | 0.229 | 0.283 | 0.061 | 0.183 | 0.156 | 1.500 | 3/3/0 |
| single_turn | bm25 | 1024 | 0.206 | 0.148 | 0.229 | 0.245 | 0.048 | 0.166 | 0.140 | 1.500 | 3/3/0 |
| single_turn | bm25 | 2048 | 0.218 | 0.306 | 0.229 | 0.326 | 0.063 | 0.193 | 0.144 | 2.000 | 1/4/1 |
| single_turn | hybrid | 512 | 0.459 | 0.155 | 0.419 | 0.272 | 0.047 | 0.176 | 0.188 | 1.667 | 3/2/1 |
| single_turn | hybrid | 1024 | 0.455 | 0.306 | 0.419 | 0.288 | 0.059 | 0.188 | 0.120 | 1.667 | 2/4/0 |
| single_turn | hybrid | 2048 | 0.275 | 0.388 | 0.419 | 0.301 | 0.062 | 0.173 | 0.131 | 2.000 | 2/2/2 |

## Paired meeting-level Lumber differences

Positive values favour Lumber. Each value is the mean of within-meeting differences.

| Comparison | Precision | Recall | ROUGE-L | BERTScore F1 | Judge |
|---|---:|---:|---:|---:|---:|
| lumber_minus_single_turn__dense__w512 | -0.125 | -0.052 | -0.003 | -0.060 | -0.167 |
| lumber_minus_turn_packed__dense__w512 | -0.062 | -0.078 | -0.006 | -0.025 | 0.000 |
| lumber_minus_word_packed__dense__w512 | -0.144 | -0.115 | 0.006 | -0.011 | 0.333 |
| lumber_minus_single_turn__dense__w1024 | -0.114 | -0.100 | 0.012 | -0.024 | 0.000 |
| lumber_minus_turn_packed__dense__w1024 | -0.033 | -0.090 | -0.012 | -0.041 | -0.167 |
| lumber_minus_word_packed__dense__w1024 | 0.021 | -0.049 | 0.010 | 0.023 | 0.500 |
| lumber_minus_single_turn__dense__w2048 | 0.076 | 0.184 | 0.014 | 0.044 | 0.167 |
| lumber_minus_turn_packed__dense__w2048 | -0.008 | 0.026 | -0.008 | -0.012 | 0.167 |
| lumber_minus_word_packed__dense__w2048 | 0.125 | 0.166 | 0.007 | 0.038 | 0.833 |
| lumber_minus_single_turn__bm25__w512 | 0.061 | 0.028 | -0.019 | 0.036 | 0.000 |
| lumber_minus_turn_packed__bm25__w512 | -0.089 | -0.031 | 0.001 | 0.027 | -0.167 |
| lumber_minus_word_packed__bm25__w512 | -0.132 | -0.064 | -0.017 | 0.010 | -0.333 |
| lumber_minus_single_turn__bm25__w1024 | 0.208 | 0.218 | 0.022 | 0.050 | 0.667 |
| lumber_minus_turn_packed__bm25__w1024 | -0.023 | 0.003 | -0.000 | 0.018 | 0.167 |
| lumber_minus_word_packed__bm25__w1024 | 0.088 | 0.099 | -0.007 | -0.005 | 0.167 |
| lumber_minus_single_turn__bm25__w2048 | 0.076 | 0.191 | -0.007 | 0.048 | 0.333 |
| lumber_minus_turn_packed__bm25__w2048 | 0.026 | 0.084 | 0.008 | 0.012 | 0.333 |
| lumber_minus_word_packed__bm25__w2048 | 0.087 | 0.176 | -0.003 | -0.009 | 0.333 |
| lumber_minus_single_turn__hybrid__w512 | 0.023 | 0.024 | 0.003 | -0.032 | -0.167 |
| lumber_minus_turn_packed__hybrid__w512 | 0.058 | 0.038 | -0.008 | 0.006 | -0.167 |
| lumber_minus_word_packed__hybrid__w512 | 0.232 | 0.070 | 0.010 | 0.016 | 0.000 |
| lumber_minus_single_turn__hybrid__w1024 | 0.066 | 0.130 | -0.005 | 0.068 | 0.333 |
| lumber_minus_turn_packed__hybrid__w1024 | 0.124 | 0.171 | -0.002 | 0.005 | 0.000 |
| lumber_minus_word_packed__hybrid__w1024 | 0.271 | 0.232 | 0.013 | 0.038 | 0.333 |
| lumber_minus_single_turn__hybrid__w2048 | 0.072 | 0.157 | 0.033 | 0.058 | 0.167 |
| lumber_minus_turn_packed__hybrid__w2048 | 0.007 | -0.015 | 0.006 | 0.014 | -0.167 |
| lumber_minus_word_packed__hybrid__w2048 | 0.023 | -0.020 | 0.045 | 0.078 | 0.333 |

## Best observed configurations

- **Retrieval recall:** `lumber__dense__w2048` (0.632)
- **ROUGE-L:** `lumber__hybrid__w2048` (0.206)
- **BERTScore F1:** `word_packed__bm25__w2048` (0.202)
- **LLM judge:** `turn_packed__hybrid__w2048` (2.333)

## Interpretation notes

- Retrieval precision and recall are word-weighted against QMSum's annotated evidence spans. First-overlap MRR structurally favours larger chunks and is diagnostic only.
- Retrieval chooses evidence under the budget, then renders selected fragments chronologically for conversational coherence.
- ROUGE and BERTScore compare generated answers with the reference answers. The judge uses the reference answer and gold transcript evidence on a 1--3 scale.
- The 1--3 judge compresses correctness, completeness, and grounding into one ordinal score; manual review remains necessary.
- The judge checkpoint is `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`, separate from the candidate checkpoints.
