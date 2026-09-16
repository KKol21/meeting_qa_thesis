# Lumber boundary-model check

**Scope:** 5 QMSum validation meetings, 38 questions, target 1000 pseudo-tokens.

This retrieval-only diagnostic changes only the Lumber boundary model. Recall is first averaged across the nine retriever x evidence-budget environments within each meeting; meetings are then weighted equally and cluster-bootstrapped.

> With only five meetings, use this to detect a large model effect or obvious failure—not to establish model equivalence.

## Headline recall

Differences are relative to `qwen2.5-14b`.

| Boundary model | Mean recall [95% CI] | Difference [95% CI] |
|---|---:|---:|
| qwen2.5-7b | 0.452 [0.265, 0.639] | -0.015 [-0.116, +0.072] |
| qwen2.5-14b | 0.468 [0.294, 0.642] | reference |
| qwen2.5-32b-bnb4 | 0.440 [0.258, 0.626] | -0.027 [-0.089, +0.011] |

## Chunk geometry

| Boundary model | Chunks | Mean words | Median words | P90 words | Mean turns |
|---|---:|---:|---:|---:|---:|
| qwen2.5-7b | 211 | 186.7 | 142.0 | 414 | 14.8 |
| qwen2.5-14b | 157 | 250.9 | 226.0 | 475 | 19.9 |
| qwen2.5-32b-bnb4 | 168 | 234.5 | 202.5 | 447 | 18.6 |

## Boundary agreement

| Models | Shared boundaries | Jaccard |
|---|---:|---:|
| qwen2.5-7b / qwen2.5-14b | 61 | 20.5% |
| qwen2.5-7b / qwen2.5-32b-bnb4 | 58 | 18.6% |
| qwen2.5-14b / qwen2.5-32b-bnb4 | 85 | 37.0% |

## Retrieval cells

| Model | Retriever | Budget | Precision | Recall | F1 | Zero-hit | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-7b | dense | 512 | 0.371 | 0.339 | 0.298 | 26.7% | 0.720 |
| qwen2.5-7b | dense | 1024 | 0.298 | 0.498 | 0.315 | 9.2% | 0.720 |
| qwen2.5-7b | dense | 2048 | 0.248 | 0.715 | 0.319 | 2.5% | 0.720 |
| qwen2.5-7b | bm25 | 512 | 0.226 | 0.198 | 0.183 | 45.8% | 0.533 |
| qwen2.5-7b | bm25 | 1024 | 0.216 | 0.351 | 0.222 | 22.5% | 0.533 |
| qwen2.5-7b | bm25 | 2048 | 0.186 | 0.568 | 0.241 | 7.5% | 0.533 |
| qwen2.5-7b | hybrid | 512 | 0.361 | 0.296 | 0.283 | 32.5% | 0.655 |
| qwen2.5-7b | hybrid | 1024 | 0.277 | 0.460 | 0.291 | 10.8% | 0.655 |
| qwen2.5-7b | hybrid | 2048 | 0.226 | 0.647 | 0.289 | 7.5% | 0.655 |
| qwen2.5-14b | dense | 512 | 0.381 | 0.279 | 0.274 | 34.2% | 0.668 |
| qwen2.5-14b | dense | 1024 | 0.336 | 0.502 | 0.343 | 20.0% | 0.668 |
| qwen2.5-14b | dense | 2048 | 0.253 | 0.680 | 0.320 | 14.2% | 0.668 |
| qwen2.5-14b | bm25 | 512 | 0.302 | 0.254 | 0.242 | 41.7% | 0.584 |
| qwen2.5-14b | bm25 | 1024 | 0.257 | 0.458 | 0.280 | 15.8% | 0.584 |
| qwen2.5-14b | bm25 | 2048 | 0.211 | 0.623 | 0.270 | 8.3% | 0.584 |
| qwen2.5-14b | hybrid | 512 | 0.335 | 0.248 | 0.251 | 34.2% | 0.650 |
| qwen2.5-14b | hybrid | 1024 | 0.325 | 0.500 | 0.330 | 17.5% | 0.650 |
| qwen2.5-14b | hybrid | 2048 | 0.236 | 0.666 | 0.303 | 10.0% | 0.650 |
| qwen2.5-32b-bnb4 | dense | 512 | 0.352 | 0.278 | 0.261 | 29.2% | 0.677 |
| qwen2.5-32b-bnb4 | dense | 1024 | 0.308 | 0.439 | 0.309 | 19.2% | 0.677 |
| qwen2.5-32b-bnb4 | dense | 2048 | 0.236 | 0.647 | 0.301 | 5.0% | 0.677 |
| qwen2.5-32b-bnb4 | bm25 | 512 | 0.260 | 0.261 | 0.225 | 41.7% | 0.591 |
| qwen2.5-32b-bnb4 | bm25 | 1024 | 0.231 | 0.391 | 0.246 | 20.0% | 0.591 |
| qwen2.5-32b-bnb4 | bm25 | 2048 | 0.208 | 0.620 | 0.269 | 8.3% | 0.591 |
| qwen2.5-32b-bnb4 | hybrid | 512 | 0.317 | 0.255 | 0.242 | 32.5% | 0.675 |
| qwen2.5-32b-bnb4 | hybrid | 1024 | 0.287 | 0.412 | 0.287 | 22.5% | 0.675 |
| qwen2.5-32b-bnb4 | hybrid | 2048 | 0.232 | 0.661 | 0.299 | 5.0% | 0.675 |

A model is promising only if its paired recall difference is meaningfully positive without obtaining that gain merely through substantially larger chunks. Confirm any promising result on the complete validation set before changing the main experiment.
