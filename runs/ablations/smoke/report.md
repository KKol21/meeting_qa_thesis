# Ablation report

**Scope:** 1 meeting(s), 6 question(s).

All per-question answers and evidence: [review.md](review.md)

> This is a smoke test. Treat rankings as pipeline validation, not experimental conclusions.

## Oracle answer-model comparison

Gold evidence is supplied here, isolating answer-model performance from retrieval.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Judge mean | Judge 1/2/3 |
|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-14b | 0.371 | 0.094 | 0.235 | 0.869 | 2.333 | 1/2/3 |
| qwen2.5-32b-bnb4 | 0.355 | 0.072 | 0.213 | 0.864 | 2.167 | 1/3/2 |
| qwen2.5-7b | 0.243 | 0.037 | 0.170 | 0.848 | 1.500 | 4/1/1 |

## Retrieval and end-to-end comparison

Means are meeting-macro averages; judge 1/2/3 counts are question totals. First-overlap MRR is retained only as a size-sensitive diagnostic.

| Chunker | Retriever | Words | Precision | Recall | First-overlap MRR | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Judge | 1/2/3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| turn_packed | dense | 512 | 0.333 | 0.092 | 0.528 | 0.283 | 0.048 | 0.175 | 0.848 | 1.667 | 3/2/1 |
| turn_packed | dense | 1024 | 0.414 | 0.327 | 0.528 | 0.346 | 0.060 | 0.211 | 0.863 | 2.000 | 1/4/1 |
| turn_packed | bm25 | 512 | 0.373 | 0.145 | 0.485 | 0.263 | 0.044 | 0.163 | 0.859 | 1.667 | 4/0/2 |
| turn_packed | bm25 | 1024 | 0.437 | 0.362 | 0.485 | 0.324 | 0.053 | 0.188 | 0.860 | 2.000 | 1/4/1 |
| turn_packed | hybrid | 512 | 0.357 | 0.123 | 0.557 | 0.275 | 0.037 | 0.167 | 0.861 | 1.667 | 2/4/0 |
| turn_packed | hybrid | 1024 | 0.397 | 0.264 | 0.557 | 0.305 | 0.052 | 0.185 | 0.862 | 2.000 | 2/2/2 |
| word_packed | dense | 512 | 0.394 | 0.186 | 0.722 | 0.272 | 0.046 | 0.173 | 0.855 | 1.167 | 5/1/0 |
| word_packed | dense | 1024 | 0.354 | 0.276 | 0.722 | 0.288 | 0.061 | 0.183 | 0.854 | 1.333 | 4/2/0 |
| word_packed | bm25 | 512 | 0.417 | 0.178 | 0.639 | 0.290 | 0.051 | 0.181 | 0.862 | 1.833 | 2/3/1 |
| word_packed | bm25 | 1024 | 0.325 | 0.267 | 0.639 | 0.325 | 0.064 | 0.194 | 0.864 | 2.000 | 1/4/1 |
| word_packed | hybrid | 512 | 0.250 | 0.109 | 0.653 | 0.249 | 0.039 | 0.170 | 0.855 | 1.500 | 3/3/0 |
| word_packed | hybrid | 1024 | 0.250 | 0.203 | 0.653 | 0.304 | 0.055 | 0.176 | 0.856 | 1.833 | 1/5/0 |
| lumber | dense | 512 | 0.273 | 0.075 | 0.436 | 0.248 | 0.027 | 0.154 | 0.848 | 1.333 | 5/0/1 |
| lumber | dense | 1024 | 0.425 | 0.329 | 0.436 | 0.300 | 0.061 | 0.188 | 0.861 | 2.000 | 2/2/2 |
| lumber | bm25 | 512 | 0.473 | 0.213 | 0.512 | 0.266 | 0.038 | 0.171 | 0.860 | 1.833 | 2/3/1 |
| lumber | bm25 | 1024 | 0.459 | 0.369 | 0.512 | 0.305 | 0.059 | 0.179 | 0.860 | 2.167 | 2/1/3 |
| lumber | hybrid | 512 | 0.460 | 0.154 | 0.550 | 0.254 | 0.039 | 0.175 | 0.855 | 1.667 | 3/2/1 |
| lumber | hybrid | 1024 | 0.382 | 0.242 | 0.550 | 0.273 | 0.048 | 0.178 | 0.858 | 1.833 | 3/1/2 |

## Paired meeting-level Lumber differences

Positive values favour Lumber. Each value is the mean of within-meeting differences.

| Comparison | Precision | Recall | ROUGE-L | BERTScore F1 | Judge |
|---|---:|---:|---:|---:|---:|
| lumber_minus_turn_packed__dense__w512 | -0.060 | -0.018 | -0.021 | -0.001 | -0.333 |
| lumber_minus_word_packed__dense__w512 | -0.120 | -0.111 | -0.019 | -0.007 | 0.167 |
| lumber_minus_turn_packed__dense__w1024 | 0.011 | 0.002 | -0.023 | -0.003 | 0.000 |
| lumber_minus_word_packed__dense__w1024 | 0.071 | 0.054 | 0.004 | 0.007 | 0.667 |
| lumber_minus_turn_packed__bm25__w512 | 0.100 | 0.068 | 0.009 | 0.001 | 0.167 |
| lumber_minus_word_packed__bm25__w512 | 0.057 | 0.035 | -0.009 | -0.002 | 0.000 |
| lumber_minus_turn_packed__bm25__w1024 | 0.022 | 0.007 | -0.009 | -0.000 | 0.167 |
| lumber_minus_word_packed__bm25__w1024 | 0.134 | 0.102 | -0.015 | -0.004 | 0.167 |
| lumber_minus_turn_packed__hybrid__w512 | 0.102 | 0.032 | 0.008 | -0.006 | 0.000 |
| lumber_minus_word_packed__hybrid__w512 | 0.210 | 0.045 | 0.005 | 0.000 | 0.167 |
| lumber_minus_turn_packed__hybrid__w1024 | -0.015 | -0.023 | -0.007 | -0.005 | -0.167 |
| lumber_minus_word_packed__hybrid__w1024 | 0.132 | 0.039 | 0.002 | 0.001 | 0.000 |

## Best observed configurations

- **Retrieval recall:** `lumber__bm25__w1024` (0.369)
- **ROUGE-L:** `turn_packed__dense__w1024` (0.211)
- **BERTScore F1:** `word_packed__bm25__w1024` (0.864)
- **LLM judge:** `lumber__bm25__w1024` (2.167)

## Interpretation notes

- Retrieval precision and recall are word-weighted against QMSum's annotated evidence spans. First-overlap MRR structurally favours larger chunks and is diagnostic only.
- Retrieval chooses evidence under the budget, then renders selected fragments chronologically for conversational coherence.
- ROUGE and BERTScore compare generated answers with the reference answers. The judge uses the reference answer and gold transcript evidence on a 1–3 scale.
- The 1–3 judge compresses correctness, completeness, and grounding into one ordinal score; manual review remains necessary.
- The judge checkpoint is `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`, separate from the candidate checkpoints.
