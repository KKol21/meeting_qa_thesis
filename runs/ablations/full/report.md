# Ablation report

**Scope:** 20 meeting(s), 142 question(s).

All per-question answers and evidence: [review.md](review.md)

## Oracle answer-model comparison

Gold evidence is supplied here, isolating answer-model performance from retrieval.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Judge mean | Judge 1/2/3 |
|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-14b | 0.360 | 0.104 | 0.228 | 0.875 | 2.533 | 5/57/80 |
| qwen2.5-32b-bnb4 | 0.362 | 0.109 | 0.225 | 0.871 | 2.586 | 12/35/95 |
| qwen2.5-7b | 0.367 | 0.107 | 0.227 | 0.872 | 2.458 | 12/56/74 |

## Retrieval and end-to-end comparison

Means are meeting-macro averages; judge 1/2/3 counts are question totals. First-overlap MRR is retained only as a size-sensitive diagnostic.

| Chunker | Retriever | Words | Precision | Recall | First-overlap MRR | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Judge | 1/2/3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| turn_packed | dense | 512 | 0.353 | 0.331 | 0.659 | 0.321 | 0.083 | 0.199 | 0.866 | 1.709 | 52/79/11 |
| turn_packed | dense | 1024 | 0.296 | 0.520 | 0.659 | 0.337 | 0.089 | 0.214 | 0.868 | 1.894 | 37/82/23 |
| turn_packed | bm25 | 512 | 0.287 | 0.271 | 0.601 | 0.309 | 0.080 | 0.197 | 0.863 | 1.645 | 56/77/9 |
| turn_packed | bm25 | 1024 | 0.235 | 0.418 | 0.601 | 0.332 | 0.086 | 0.206 | 0.866 | 1.815 | 41/86/15 |
| turn_packed | hybrid | 512 | 0.352 | 0.338 | 0.682 | 0.323 | 0.087 | 0.203 | 0.867 | 1.793 | 45/83/14 |
| turn_packed | hybrid | 1024 | 0.283 | 0.483 | 0.682 | 0.333 | 0.087 | 0.208 | 0.867 | 1.983 | 33/78/31 |
| word_packed | dense | 512 | 0.365 | 0.354 | 0.720 | 0.321 | 0.078 | 0.198 | 0.865 | 1.787 | 44/83/15 |
| word_packed | dense | 1024 | 0.290 | 0.505 | 0.720 | 0.328 | 0.083 | 0.203 | 0.867 | 1.952 | 38/77/27 |
| word_packed | bm25 | 512 | 0.285 | 0.267 | 0.616 | 0.317 | 0.082 | 0.198 | 0.864 | 1.684 | 53/79/10 |
| word_packed | bm25 | 1024 | 0.225 | 0.406 | 0.616 | 0.323 | 0.082 | 0.204 | 0.866 | 1.849 | 42/79/21 |
| word_packed | hybrid | 512 | 0.344 | 0.323 | 0.729 | 0.315 | 0.077 | 0.195 | 0.864 | 1.763 | 47/82/13 |
| word_packed | hybrid | 1024 | 0.274 | 0.474 | 0.729 | 0.337 | 0.087 | 0.209 | 0.869 | 1.904 | 36/84/22 |
| lumber | dense | 512 | 0.377 | 0.368 | 0.659 | 0.327 | 0.082 | 0.202 | 0.867 | 1.806 | 43/82/17 |
| lumber | dense | 1024 | 0.302 | 0.526 | 0.659 | 0.338 | 0.088 | 0.209 | 0.868 | 1.945 | 31/84/27 |
| lumber | bm25 | 512 | 0.299 | 0.277 | 0.582 | 0.312 | 0.082 | 0.200 | 0.864 | 1.681 | 57/66/19 |
| lumber | bm25 | 1024 | 0.237 | 0.413 | 0.582 | 0.329 | 0.087 | 0.206 | 0.867 | 1.879 | 43/74/25 |
| lumber | hybrid | 512 | 0.356 | 0.332 | 0.663 | 0.328 | 0.088 | 0.207 | 0.867 | 1.769 | 46/79/17 |
| lumber | hybrid | 1024 | 0.289 | 0.493 | 0.663 | 0.339 | 0.090 | 0.206 | 0.868 | 1.911 | 34/87/21 |

## Paired meeting-level Lumber differences

Positive values favour Lumber. Each value is the mean of within-meeting differences.

| Comparison | Precision | Recall | ROUGE-L | BERTScore F1 | Judge |
|---|---:|---:|---:|---:|---:|
| lumber_minus_turn_packed__dense__w512 | 0.024 | 0.037 | 0.003 | 0.001 | 0.097 |
| lumber_minus_word_packed__dense__w512 | 0.012 | 0.014 | 0.004 | 0.002 | 0.020 |
| lumber_minus_turn_packed__dense__w1024 | 0.006 | 0.006 | -0.005 | -0.001 | 0.051 |
| lumber_minus_word_packed__dense__w1024 | 0.012 | 0.021 | 0.006 | 0.000 | -0.007 |
| lumber_minus_turn_packed__bm25__w512 | 0.012 | 0.007 | 0.004 | 0.001 | 0.036 |
| lumber_minus_word_packed__bm25__w512 | 0.014 | 0.010 | 0.003 | 0.000 | -0.003 |
| lumber_minus_turn_packed__bm25__w1024 | 0.002 | -0.004 | -0.000 | 0.000 | 0.064 |
| lumber_minus_word_packed__bm25__w1024 | 0.012 | 0.007 | 0.002 | 0.001 | 0.030 |
| lumber_minus_turn_packed__hybrid__w512 | 0.005 | -0.007 | 0.004 | 0.001 | -0.024 |
| lumber_minus_word_packed__hybrid__w512 | 0.012 | 0.009 | 0.012 | 0.003 | 0.006 |
| lumber_minus_turn_packed__hybrid__w1024 | 0.006 | 0.010 | -0.001 | 0.001 | -0.073 |
| lumber_minus_word_packed__hybrid__w1024 | 0.015 | 0.019 | -0.002 | -0.001 | 0.007 |

## Best observed configurations

- **Retrieval recall:** `lumber__dense__w1024` (0.526)
- **ROUGE-L:** `turn_packed__dense__w1024` (0.214)
- **BERTScore F1:** `word_packed__hybrid__w1024` (0.869)
- **LLM judge:** `turn_packed__hybrid__w1024` (1.983)

## Interpretation notes

- Retrieval precision and recall are word-weighted against QMSum's annotated evidence spans. First-overlap MRR structurally favours larger chunks and is diagnostic only.
- Retrieval chooses evidence under the budget, then renders selected fragments chronologically for conversational coherence.
- ROUGE and BERTScore compare generated answers with the reference answers. The judge uses the reference answer and gold transcript evidence on a 1–3 scale.
- The 1–3 judge compresses correctness, completeness, and grounding into one ordinal score; manual review remains necessary.
- The judge checkpoint is `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`, separate from the candidate checkpoints.
