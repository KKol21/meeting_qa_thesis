# Validation and sensitivity analyses

## Scope

These analyses use the same 20-meeting QMSum validation subset containing 142
questions. They support parameter selection and interpretation; they are not
independent confirmatory tests. Unless stated otherwise, retrieval is evaluated
with word-weighted precision, recall, F1, zero-hit rate, and first-overlap MRR
against QMSum's turn-level evidence annotations.

The analyses cover:

1. Lumber target-window selection;
2. deterministic baseline chunk-size sensitivity;
3. evidence-clipping sensitivity;
4. a boundary-shuffled Lumber control.

No answers were generated for these sensitivity analyses. The complete result
tables and question-level JSON remain the authoritative numerical records.

## Main conclusions

| Question | Finding | Decision |
|---|---|---|
| Which Lumber target should be used? | No target dominates every retrieval budget. Target 1,000 gives 243 median words, the closest match to the deterministic baselines, and competitive mean F1. | Freeze Lumber target 1,000 before held-out testing. |
| Does the 256-word baseline choice determine the result? | Moving to 128 or 512 changes retrieval modestly, with clear interactions with retriever and evidence budget. Larger chunks slightly improve some averages but increase zero-hit. | Retain 256 words as the balanced deterministic baseline. |
| Does clipping create the observed result? | Clipping affects recall most at the 512-word evidence budget. F1 changes are generally small, especially at 1,024 and 2,048 words. | Retain exact-budget clipping and document it explicitly. |
| Do Lumber's semantic boundary locations matter? | Real boundaries modestly improve F1 over geometry-matched shuffled boundaries in several conditions, but recall and zero-hit improvements are not consistent. | Treat boundary placement as a supported but configuration-dependent secondary finding. |
| Which evidence budget is the main operating point? | Larger budgets raise recall, but answer-quality gains saturate and sometimes reverse. | Use 1,024 words as the primary budget; retain 512 and 2,048 as sensitivity conditions. |

## 1. Lumber target-window sweep

The [Lumber sweep](../runs/ablations/lumber-sweep/sweep.md) compares targets
500, 750, 1,000, 1,250, and 1,500 while holding the boundary model, prompt,
decoding, retrievers, and evidence budgets fixed. `target_tokens` is Lumber's
rendered-word approximation rather than the boundary model's tokenizer count.

Increasing the target produced progressively larger and fewer chunks:

| Target | Chunks | Median words | Mean turns |
|---:|---:|---:|---:|
| 500 | 1,033 | 162.0 | 10.8 |
| 750 | 800 | 204.0 | 13.9 |
| 1,000 | 651 | 243.0 | 17.1 |
| 1,250 | 550 | 264.5 | 20.2 |
| 1,500 | 484 | 288.5 | 23.0 |

Across all retrieval conditions, target 750 obtained the highest mean recall
(0.490), while target 1,000 obtained the highest mean F1 (0.299). Performance
depended on evidence budget: smaller targets were generally safer at 512 words,
whereas larger targets became more competitive at 2,048 words. Target 1,500
had the weakest overall recall and the highest zero-hit rate.

The predeclared selection rule considered targets within 0.01 recall of the
best and selected the median chunk length closest to the deterministic baseline
medians. It therefore selected **1,000**, whose 243-word median was only five
words from the baseline midpoint.

Adjacent target settings shared only 48--54% of their combined boundaries.
The target parameter therefore changes more than nominal chunk length: it also
changes which topical transitions the model selects. This makes freezing the
chosen value before test evaluation particularly important.

## 2. Deterministic chunk-size sensitivity

The [baseline sweep](../runs/ablations/baseline-sweep/sweep.md) evaluates
turn-packed and strict word-packed chunks at 128, 256, and 512 words. The table
below shows the primary dense/1,024-word condition.

| Chunker | Size | Precision | Recall | F1 | Zero-hit |
|---|---:|---:|---:|---:|---:|
| Turn-packed | 128 | 0.284 | 0.488 | 0.303 | 8.9% |
| Turn-packed | 256 | 0.294 | 0.487 | 0.311 | 14.4% |
| Turn-packed | 512 | 0.300 | 0.500 | 0.323 | 24.2% |
| Word-packed | 128 | 0.286 | 0.507 | 0.310 | 7.8% |
| Word-packed | 256 | 0.290 | 0.491 | 0.311 | 18.5% |
| Word-packed | 512 | 0.313 | 0.504 | 0.331 | 25.4% |

The 512-word setting sometimes improves mean F1, especially with a larger
evidence budget, but it also produces substantially more zero-hit questions.
The 128-word setting reduces zero-hit but tends to weaken BM25 and aggregate
F1. Consequently, 256 is not uniquely optimal, but it is a defensible
compromise and the main conclusions are not artifacts of that single value.

MRR generally rises with chunk size. This should not be interpreted as an
independent quality gain because a larger chunk is mechanically more likely to
overlap at least one annotated evidence turn.

## 3. Evidence-clipping sensitivity

The [clipping analysis](../runs/ablations/full/clipping-sensitivity.md) holds
rankings fixed and compares three policies:

- `clip`: include exactly the available word budget, clipping the final chunk;
- `drop_partial`: stop before the final chunk if it would be partial;
- `expand_partial`: include that chunk in full and permit budget overflow.

For Lumber with dense retrieval in the existing 550-target full validation
run:

| Budget | Clip recall/F1 | Drop change in recall/F1 | Expand change in recall/F1 | Mean underfill/overflow |
|---:|---:|---:|---:|---:|
| 512 | 0.350/0.310 | -0.068/-0.022 | +0.050/+0.015 | 131/111 words |
| 1,024 | 0.535/0.332 | -0.035/-0.001 | +0.035/-0.001 | 113/127 words |
| 2,048 | 0.686/0.292 | -0.003/+0.008 | +0.009/-0.005 | 133/91 words |

Clipping is nearly universal for Lumber and turn-packed chunks because their
sizes do not divide the evidence budgets evenly. It is uncommon for strict
word-packed chunks because 512, 1,024, and 2,048 are multiples of the 256-word
chunk size.

The policy materially changes recall at the smallest budget, but F1 differences
are usually small. Dropping the partial chunk wastes capacity, while expanding
it violates equal evidence budgets. Exact clipping is therefore retained as
the fairest primary policy. The analysis does not test downstream answer
generation, and its Lumber rows characterize the earlier 550-target run rather
than the subsequently selected 1,000-target segmentation.

## 4. Boundary-shuffled Lumber control

The corrected [boundary control](../runs/ablations/boundary-control/control.md)
uses the selected **1,000-target** Lumber segmentation. For each meeting, ten
controls permute Lumber's exact turns-per-chunk distribution across legal turn
boundaries. From 1,000 random permutations per control, the partition with the
closest word-size distribution is retained. Turns remain complete, ordered,
and present exactly once.

The geometry match is close:

| Property | Lumber | Shuffled control |
|---|---:|---:|
| Mean words | 277.4 | 277.4 |
| Median words | 243 | 233 |
| Mean turns | 17.1 | 17.1 |
| Median turns | 13 | 13 |

Only 12.8% of actual Lumber boundaries are retained on average, indicating
that the semantic locations were substantially disrupted.

Selected paired results are shown below. Confidence intervals are
meeting-level paired bootstrap intervals for actual Lumber minus the mean of
the ten shuffled controls.

| Retriever | Budget | Recall difference [95% CI] | F1 difference [95% CI] |
|---|---:|---:|---:|
| Dense | 512 | +0.030 [-0.007, +0.069] | **+0.036 [+0.005, +0.068]** |
| Dense | 1,024 | +0.025 [-0.013, +0.062] | **+0.027 [+0.003, +0.049]** |
| Dense | 2,048 | +0.010 [-0.028, +0.047] | +0.007 [-0.006, +0.021] |
| BM25 | 512 | **+0.045 [+0.003, +0.086]** | **+0.045 [+0.009, +0.080]** |
| BM25 | 1,024 | +0.035 [-0.017, +0.086] | +0.019 [-0.008, +0.047] |
| BM25 | 2,048 | +0.025 [-0.013, +0.059] | **+0.012 [+0.001, +0.022]** |
| Hybrid | 512 | +0.005 [-0.033, +0.043] | +0.013 [-0.017, +0.042] |
| Hybrid | 1,024 | +0.031 [-0.009, +0.070] | +0.019 [-0.001, +0.039] |
| Hybrid | 2,048 | +0.009 [-0.034, +0.052] | +0.010 [-0.004, +0.025] |

Actual Lumber has positive mean recall and F1 differences in all nine
conditions, but many intervals include zero. The strongest evidence concerns
F1 for dense retrieval at 512 and 1,024 words and both recall and F1 for BM25
at 512 words. This supports the claim that boundary placement sometimes adds
value beyond chunk geometry, but not that semantic boundaries universally
improve retrieval.

Zero-hit rates do not consistently favour Lumber. For example, dense/1,024
has 21.5% zero-hit for Lumber versus 19.5% for the shuffled control, despite
Lumber's higher mean recall and F1. The semantic segmentation can therefore
improve the quality of successful retrieval while still producing occasional
complete misses.

First-overlap MRR also does not favour Lumber consistently: its paired
difference is negative for dense and hybrid retrieval. This reinforces the
decision to treat MRR as a secondary, size-sensitive diagnostic.

## Relation to the main validation run

The earlier [full ablation report](../runs/ablations/full/report.md) used the
550-target Lumber segmentation, before target 1,000 was selected. In that run,
Lumber with dense retrieval and a 1,024-word evidence budget produced the best
LLM-judge mean (2.032) and BERTScore F1 (0.228). Lumber with dense retrieval at
2,048 words produced the highest retrieval recall (0.686), while Lumber with
hybrid retrieval at 2,048 words produced the highest ROUGE-L (0.216).

This pattern motivates 1,024 words as the primary operating point: increasing
the budget to 2,048 improves evidence coverage but does not improve the best
answer-quality results. Because the run used target 550, its answer results
should not be presented as a held-out evaluation of the subsequently selected
1,000-target configuration.

## Integrated interpretation

The combined evidence separates three effects:

1. **Evidence budget has a large effect.** Increasing the budget reliably
   raises recall, but downstream answer gains saturate.
2. **Chunk geometry matters.** Very small chunks lower catastrophic misses in
   some conditions, while larger chunks sometimes improve average precision or
   F1 but increase zero-hit.
3. **Boundary location contributes additional information.** After closely
   matching geometry, actual Lumber boundaries retain modest F1 advantages in
   several conditions. These advantages are smaller and less consistent than
   the effect of evidence budget.

The evidence therefore supports a restrained thesis claim: semantic boundary
placement can improve retrieval beyond size alone, particularly under dense
retrieval and constrained context, but its benefit is configuration-dependent.

## Final experimental decisions

- Lumber target: **1,000 pseudo-tokens**.
- Deterministic chunk size: **256 words**.
- Primary evidence budget: **1,024 words**.
- Sensitivity evidence budgets: **512 and 2,048 words**.
- Primary retriever: **dense**; BM25 and hybrid remain ablations.
- Evidence selection: **exact-budget clipping**.
- Boundary-shuffled controls: **secondary retrieval analysis only**.

No additional broad validation sweeps are necessary. The next methodological
step should be to freeze these decisions and run the held-out test evaluation
once. Repeating analyses after observing test results would weaken the
validation/test separation.

## Limitations

- All selections and sensitivity checks reuse one 20-meeting validation subset.
- Boundary-control intervals are exploratory and are not corrected for nine
  comparisons.
- The shuffled control exactly preserves chunk count and turns-per-chunk
  distribution, but matches word sizes approximately; its median is ten words
  below Lumber's.
- QMSum relevance is annotated at turn level, so every word in a gold turn is
  counted as relevant.
- MRR remains structurally sensitive to chunk size.
- The clipping alternatives either leave the budget underfilled or permit
  overflow and therefore do not offer perfectly equivalent comparisons.
- Retrieval-only sensitivity results do not directly establish effects on
  generated answers.
