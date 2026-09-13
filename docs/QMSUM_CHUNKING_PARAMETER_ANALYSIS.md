# QMSum evidence and turn-length analysis

## Purpose

This analysis checks whether the experiment's 256-word baseline chunks,
550-pseudo-token Lumber window, and 512/1,024/2,048-word evidence budgets are
reasonable for QMSum. It is descriptive, not evidence that these values are
optimal.

The parameter interpretation is based primarily on QMSum training data. The
20-meeting validation subset used by the current experiment and the untouched
test split are shown as checks. Test statistics were not used to select a
parameter.

## Data and definitions

Only QMSum's specific-query questions are included, matching
`qmsum.load_meeting()`. The local copy contains:

| Scope | Meetings | Turns | Questions | Annotated spans |
|---|---:|---:|---:|---:|
| Training split | 162 | 88,336 | 1,095 | 1,206 |
| Experimental validation subset | 20 | 11,128 | 142 | 148 |
| Test split | 35 | 20,718 | 244 | 286 |

A *word* is one whitespace-separated item in transcript content. Speaker labels
and turn IDs are excluded from word-based chunk and evidence budgets. An
annotated span is one inclusive QMSum `(start_turn, end_turn)` range. Per-question
gold evidence is the union of all annotated ranges, so overlapping turns are
counted once.

Lumber's value called `target_tokens` is different: `estimate_tokens()` returns
`round(1.2 * whitespace words)` over rendered turns, including IDs and speaker
labels. It is an approximation from the LumberChunker procedure, not a model
tokenizer count. The two units must not be presented as interchangeable.

Reported percentiles use the nearest-rank definition. Standard deviations are
sample standard deviations.

## Turn lengths

Transcript turns are short but strongly right-skewed.

| Split | Mean (SD) words | P50 | P90 | P95 | P99 | Maximum | Turns over 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 16.4 (34.1) | 6 | 37 | 62 | 156 | 1,810 | 272 (0.31%) |
| Validation subset | 16.2 (32.4) | 6 | 38 | 66 | 156 | 725 | 28 (0.25%) |
| Test | 16.0 (30.8) | 7 | 36 | 60 | 152 | 756 | 50 (0.24%) |

The validation subset closely follows the training distribution. A 256-word
limit accommodates more than 99.6% of individual turns without splitting, but
rare monologues are much longer. This distinction explains why the turn-packed
baseline is a *soft* limit.

The single-turn baseline uses this distribution directly: its median chunk is
6 words and its mean is about 16 words, with the same long tail shown above.
It is therefore a deliberately fine-grained, turn-atomic retrieval baseline,
not a size-matched comparison with the 256-word or Lumber conditions.

Across training meetings, 256-word turn-packed chunks contain a mean of 229
words and 14 complete turns; the medians are 242 words and 13 turns. The 95th
percentile is 256 words, but 272 of 6,318 chunks (4.3%) exceed the limit because
a single long turn is never split. The maximum is 1,810 words.

The strict word-packed baseline produces 5,733 training chunks with a mean of
252 words and a median of 256. No chunk exceeds 256 words because long turns
are split and the speaker attribution is repeated in the next chunk. Together,
the non-semantic baselines contrast turn atomicity, packing while preserving
turns, and exact size control.

## Gold evidence spans

### Individual annotated ranges

| Split | Mean (SD) words | P50 | P75 | P90 | P95 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Training | 783 (911) | 454 | 947 | 1,938 | 2,657 | 7,470 |
| Validation subset | 837 (971) | 475 | 862 | 2,028 | 2,955 | 4,834 |
| Test | 736 (826) | 425 | 849 | 1,825 | 2,318 | 5,113 |

The median training span covers 21 turns; P75 is 48, P90 is 126, and P95 is
199 turns. Only 26.0% of training spans contain at most 256 content words.
Consequently, a 256-word chunk is intentionally smaller than most annotated
evidence spans. A typical answer therefore requires retrieval across more than
one chunk rather than assuming that one chunk contains the complete gold span.

### Per-question evidence union

| Split | Mean (SD) words | P50 | P75 | P90 | P95 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Training | 862 (961) | 511 | 1,036 | 2,068 | 2,874 | 7,470 |
| Validation subset | 873 (986) | 530 | 905 | 2,051 | 2,955 | 4,834 |
| Test | 862 (960) | 480 | 1,107 | 1,988 | 2,632 | 7,352 |

The large mean--median difference and high upper percentiles show a substantial
long tail. Reporting only the mean would overstate the evidence required by a
typical question while hiding difficult cases.

Most questions have one contiguous annotation. Multiple spans occur for 88 of
1,095 training questions (8.0%), 6 of 142 selected validation questions (4.2%),
and 38 of 244 test questions (15.6%). Among fragmented training questions, gold
evidence occupies a median of 49.1% of the content between the first and last
gold turn, and the median inter-span gap is 28 turns. Thus, combining all turns
between separated annotations would often add substantial irrelevant content.

## Interpretation of the chosen values

### Baseline chunk size: 256 words

The 256-word setting is defensible as a controlled retrieval unit because:

- more than 99.6% of turns fit within it;
- the turn-packed implementation creates chunks of about 14 turns on average;
- the strict implementation gives an exact, nearly fully utilized size control;
- the median gold evidence union is approximately two such chunks, making the
  retrieval task non-trivial without making the units extremely small.

It should not be claimed that 256 words matches the natural evidence-span size.
It deliberately does not: 74.0% of training spans are longer. The thesis should
also disclose the soft-limit tail of the turn-packed baseline.

### Lumber window: 550 pseudo-tokens

On training data, an individual annotated range has a median of 649 Lumber
pseudo-tokens. Only 42.3% of individual spans and 37.4% of complete per-question
gold unions are at most 550 pseudo-tokens. At a target of 1,000, these shares
increase to 66.4% and 62.5%, respectively.

| Scope | Individual span <= 550 | <= 1,000 | Gold union <= 550 | <= 1,000 |
|---|---:|---:|---:|---:|
| Training | 42.3% | 66.4% | 37.4% | 62.5% |
| Validation subset | 40.5% | 64.2% | 38.0% | 62.7% |
| Test | 44.8% | 68.2% | 38.1% | 62.7% |

These comparisons establish scale, not a requirement that one Lumber window
contain an entire answer span. Lumber predicts a local semantic boundary; it is
not given the question or gold evidence. The strongest justification for 550
should therefore be that it is the inherited LumberChunker setting used for the
primary adaptation, while the QMSum statistics show that it produces relatively
local context.

A one-meeting sensitivity check on `Bed002`, holding the 14B boundary model
fixed, supports including 1,000 as a small ablation. The 550 setting produced 70
chunks (69 internal boundaries), whereas 1,000 produced 48 chunks (47
boundaries). Of the 47 boundaries under 1,000, 37 also occurred exactly under
550: 78.7% retention, 53.6% in the reverse direction, and 46.8% Jaccard overlap.
Mean chunk length increased from 205 to 299 content words. This suggests that
the larger window removes some fine-grained boundaries while preserving many
boundary locations, but the result is exploratory because it uses one meeting
and sequential decisions are not independent.

### Retrieved-evidence budgets: 512, 1,024, and 2,048 words

The budgets correspond to approximately two, four, and eight strict 256-word chunks.
In training data, 50.2% of questions contain at most 512 gold words and 74.7%
contain at most 1,024; 89.6% contain at most 2,048. The corresponding
validation-subset shares are 48.6%, 78.9%, and 89.4%; test shares are 52.0%,
73.8%, and 91.4%.

This makes 512 a median-scale condition, 1,024 a broader condition near the
training P75, and 2,048 a high-context condition near P90. None can recover
every gold word for all long-tail questions, even with perfect ranking. Actual
recall can be lower because retrieved chunks also contain non-gold boundary
content. `select_evidence()` clips the last selected turn fragment, so the
budgets themselves are hard word limits.

## Recommended methodological claim

The data support the following restrained justification:

> The 256-word baseline size retains complete turns in more than 99.6% of cases
> while yielding retrieval units of approximately 14 turns. It is evaluated in
> both soft turn-preserving and strict word-packed forms. Evidence budgets of
> 512, 1,024, and 2,048 words represent approximately two, four, and eight
> baseline chunks and probe median, upper-quartile, and near-P90 gold evidence
> sizes. The 550-token
> Lumber window follows the adapted method's approximate token accounting and
> is treated as a method parameter rather than as a value optimized on QMSum.

If time permits, the 1,000-target Lumber condition is the most informative
additional chunking ablation. It tests whether conclusions depend on the
primary method's relatively local window. It should be run across all selected
validation meetings before its boundary stability is presented as more than a
single-meeting observation.

## Limitations

- Whitespace words and Lumber pseudo-tokens are reproducible accounting units,
  not linguistic words or model-tokenizer tokens.
- QMSum marks relevant turn ranges, not topic boundaries. Their lengths cannot
  identify an optimal semantic chunk size.
- Range-based relevance treats every word in a labeled turn as relevant, even
  when only part of the turn supports the answer.
- The selected validation subset contains only 20 meetings and 142 questions.
- The distributions are highly skewed; means should always be accompanied by
  medians or upper percentiles.
- The 1,000-target comparison currently covers only `Bed002`.
