# Qualitative analysis: recall versus answer metrics

This review asks why low evidence recall can coexist with near-oracle ROUGE or
BERTScore, and whether higher recall reliably produces better answers. It is
exploratory: it diagnoses metric behavior but does not replace the full
meeting-level comparison.

## Comparison and selection rule

- Dataset: the version-2 full run, 20 QMSum validation meetings and 142
  questions.
- Retrieved condition: `lumber__dense__w512`.
- Control: `oracle-14b`.
- Both answer paths use the same Qwen2.5-14B model, prompt, and greedy
  generation. Only the evidence changes.
- The retrieved condition was selected reproducibly with
  `random.Random(42).choice(sorted(all_18_conditions))`; it was not selected
  for having unusually good or bad results.
- Cases below were selected post hoc to illustrate diagnostic patterns: similar
  BERTScore with different completeness, zero-recall lexical overlap,
  low-recall apparent success, high-recall failure, and a suspected annotation
  error.

Primary artifacts:

- [retrieval summary](../runs/ablations/full/retrieval/summary.json)
- [retrieved-answer evaluation](../runs/ablations/full/evaluation/retrieval-14b.json)
- [oracle-14B evaluation](../runs/ablations/full/evaluation/oracle-14b.json)
- [full generated review](../runs/ablations/full/review.md)

## Aggregate result

The primary report values are equal-weight meeting averages.

| Evidence | Precision | Retrieved recall | ROUGE-L | BERTScore F1 | Judge mean |
|---|---:|---:|---:|---:|---:|
| Lumber + dense, 512 words | 0.377 | 0.368 | 0.202 | 0.867 | 1.806 |
| Annotated oracle, uncapped | — | not measured; all labelled evidence supplied | 0.228 | 0.875 | 2.533 |

ROUGE-L falls by only 0.026 and BERTScore by 0.008, while the judge falls by
0.727. The overlap metrics therefore compress a much larger difference in
answer completeness and correctness.

The BERTScores are not actually identical. For this condition, the unrounded
question-average F1 is 0.867811 versus 0.875358 for oracle. Among the 142
retrieved answers, 139 values remain distinct at six decimal places. The
appearance of sameness comes from a narrow raw range, aggregation, and display
rounding.

## Question-level relationship with recall

The following are unweighted question-level diagnostics for the selected
condition. They are not the meeting-macro estimands used for the main claims.

| Metric | Pearson correlation with recall |
|---|---:|
| Retrieval precision | 0.524 |
| ROUGE-1 F1 | 0.401 |
| ROUGE-L F1 | 0.239 |
| BERTScore F1 | 0.461 |
| LLM-judge score | 0.553 |

| Recall interval | Questions | Precision | ROUGE-L | BERTScore F1 | Judge |
|---|---:|---:|---:|---:|---:|
| exactly 0 | 30 | 0.000 | 0.183 | 0.855 | 1.133 |
| (0, 0.25) | 40 | 0.418 | 0.195 | 0.863 | 1.800 |
| [0.25, 0.50) | 22 | 0.462 | 0.229 | 0.875 | 1.955 |
| [0.50, 0.75) | 23 | 0.586 | 0.222 | 0.872 | 2.087 |
| [0.75, 1.00] | 27 | 0.573 | 0.217 | 0.880 | 2.259 |

Recall is meaningful: answers judged 1, 2, and 3 have mean recalls of 0.126,
0.425, and 0.736 respectively. It is not sufficient, however. ROUGE-L is not
monotonic across the upper recall bins, and zero-recall answers retain a high
0.183 mean ROUGE-L.

Three design facts explain part of this behavior:

1. [`score_evidence()`](../src/meeting_qa_chunking/evidence.py#L167-L193)
   measures words recovered from entire gold-labelled turns, not answer facts.
   The gold spans average 873 words, have a median of 533, and exceed the
   512-word budget for 73/142 questions.
2. [`score_answer()`](../src/meeting_qa_chunking/answering.py#L18-L24) gives
   ROUGE only the candidate and reference. Topic words can overlap even when
   the retrieved passage supports the wrong answer.
3. [`add_bertscore()`](../src/stages/ablation_evaluate.py#L80-L93) likewise
   compares only candidate and reference. Baseline rescaling is disabled
   ([`ablation_evaluate.py:249-259`](../src/stages/ablation_evaluate.py#L249-L259)),
   leaving a high common similarity floor. It measures semantic resemblance,
   not grounding or complete fact coverage.

Chunk size is an additional confound: Lumber chunks average 167 words in this
run, compared with 226 for turn-packed and 252 for word-packed. The observed
relationship therefore concerns each complete chunking configuration, not
semantic boundaries in isolation.

## Case 1: nearly identical BERTScore, different completeness

**`education_18`, question 3:** decision not to accredit the University of
South Wales for teacher training.

| | Recall | Precision | ROUGE-L | BERTScore F1 | Judge |
|---|---:|---:|---:|---:|---:|
| Retrieved | 0.556 | 0.801 | 0.231 | **0.864688** | 2 |
| Oracle | all labelled evidence | — | 0.243 | **0.864160** | 3 |

The retrieved answer correctly covers the independent accreditation process,
effects on students and staff, institutional responsibility, and the appeal.
It omits the later discussion of geographical provision and postgraduate
accessibility. The oracle answer includes those points. BERTScore is
effectively unchanged—and is marginally higher for the incomplete answer—while
the judge detects the omission.

**Interpretation:** token-level semantic alignment is dominated by the many
shared entities and propositions. A missing subtopic has little effect on the
average matching score. This is the clearest reason not to interpret a
0.001-level BERTScore difference as equal completeness.

Artifacts: [retrieval](../runs/ablations/full/retrieval/education_18.json),
[retrieved answer](../runs/ablations/full/answers/retrieval-14b/education_18.json),
[oracle answer](../runs/ablations/full/answers/oracle-14b/education_18.json).

## Case 2: zero recall but almost identical ROUGE-L

**`ES2009b`, question 2:** functions and buttons of the remote control.

| | Recall | Precision | ROUGE-L | BERTScore F1 | Judge |
|---|---:|---:|---:|---:|---:|
| Retrieved | **0.000** | 0.000 | **0.182** | 0.860 | 1 |
| Oracle | all labelled evidence | — | **0.181** | 0.873 | 2 |

The annotated evidence concerns voice control for locating a lost remote, a
200-foot range, and a television-mounted sticky pad/page button. Retrieval
instead selects discussion of ordinary power, channel, and volume functions.
The retrieved answer is therefore wrong for the reference despite repeatedly
using words such as *remote control*, *functions*, and *buttons*. Those shared
topic words are enough for almost the same ROUGE-L as the oracle answer.

The oracle is not perfect either: it covers voice-controlled location but
omits the sticky pad, so the judge assigns 2 rather than 3. A low oracle ROUGE
ceiling makes the retrieved-oracle gap look smaller still.

Artifacts: [retrieval](../runs/ablations/full/retrieval/ES2009b.json),
[retrieved answer](../runs/ablations/full/answers/retrieval-14b/ES2009b.json),
[oracle answer](../runs/ablations/full/answers/oracle-14b/ES2009b.json).

## Case 3: low numerical recall can contain a concentrated answer

**`IS1006c`, question 6:** trend watching and appearance design.

| | Recall | Precision | Gold/retrieved words | ROUGE-L | BERTScore F1 | Judge |
|---|---:|---:|---:|---:|---:|---:|
| Retrieved | **0.250** | **1.000** | 2051 / 512 | 0.188 | 0.853 | 3 |
| Oracle | all labelled evidence | — | 2051 / 2051 | 0.206 | 0.871 | 3 |

Every retrieved word belongs to a gold-labelled turn, but the annotation spans
2051 words. The 512-word budget therefore cannot exceed 0.250 recall. The
selected material contains the central discussion of trend watching and the
fruit-and-vegetable appearance, allowing a broadly relevant answer.

Manual review is less generous than the judge: the retrieved answer omits
several reference details, including fancy/identifiable design,
technological innovation, ease of use, sponginess, and the rubber discussion.
A score of 2 appears more defensible than the saved 3.

**Interpretation:** low word recall can reflect a large, redundant gold span
rather than total evidence failure. It also shows that the LLM judge can
over-credit a fluent, topically correct answer.

Artifacts: [retrieval](../runs/ablations/full/retrieval/IS1006c.json),
[retrieved answer](../runs/ablations/full/answers/retrieval-14b/IS1006c.json),
[oracle answer](../runs/ablations/full/answers/oracle-14b/IS1006c.json).

## Case 4: full recall is not enough when precision is low

**`TS3010b`, question 5:** discussion of new features.

| | Recall | Precision | Gold/retrieved words | ROUGE-L | BERTScore F1 | Judge |
|---|---:|---:|---:|---:|---:|---:|
| Retrieved | **1.000** | **0.301** | 154 / 512 | 0.144 | 0.870 | 1 |
| Oracle | all labelled evidence | — | 154 / 154 | 0.333 | 0.908 | 2 |

Retrieval includes all annotated words about LCD cost, speech recognition, and
revisiting the decision, but surrounds them with roughly 358 non-gold words
about power buttons, channels, and teletext. The generated answer mixes those
distractors into the response and overstates that speech recognition was
definitively rejected. The oracle answer stays focused, although it omits the
plan to return to the issue.

**Interpretation:** recall guarantees inclusion, not salience. Precision,
evidence ordering, and generation behavior determine whether the model uses
the included evidence correctly. The saved judge score of 1 may be harsh—the
retrieved answer contains some correct core information—so this case also
illustrates judge noise.

Artifacts: [retrieval](../runs/ablations/full/retrieval/TS3010b.json),
[retrieved answer](../runs/ablations/full/answers/retrieval-14b/TS3010b.json),
[oracle answer](../runs/ablations/full/answers/oracle-14b/TS3010b.json).

## Case 5: apparent QMSum span error reverses the comparison

**`TS3010c`, question 5:** Marketing's market-trend findings.

| | Recall | Precision | ROUGE-L | BERTScore F1 | Judge |
|---|---:|---:|---:|---:|---:|
| Retrieved | **0.000** | 0.000 | **0.383** | **0.896** | 2 |
| Oracle | all labelled evidence | — | **0.026** | **0.825** | 1 |

The raw QMSum record labels turns 147–162 as relevant. Locally, those turns
contain short concept-design exchanges. The reference facts—young/trendy
versus old/rich, fruit and vegetables, sponginess, dark colours, and wood—are
stated in turns 133 and 135. Retrieval selects those semantically correct
turns, but they are counted as non-gold, producing zero recall. Oracle evidence
uses the labelled 147–162 range, causing the oracle model to report
insufficient evidence.

This is not evidence that zero recall can outperform oracle. It is an apparent
annotation mismatch in the checked-in QMSum file. Because the same span drives
recall, oracle evidence, and judge context, one label error distorts all three.

The raw record is `data/raw/qmsum/data/ALL/val/TS3010c.json` (local data,
ignored by Git). Related artifacts: [retrieval](../runs/ablations/full/retrieval/TS3010c.json),
[retrieved answer](../runs/ablations/full/answers/retrieval-14b/TS3010c.json),
[oracle answer](../runs/ablations/full/answers/oracle-14b/TS3010c.json).

Do not silently remove or relabel this case after seeing its outcome. A
defensible response is to define a fixed audit rule for all suspicious cases,
retain the original analysis, and report any corrected-label analysis as a
transparent sensitivity check.

## Conclusions for review

1. Recall is related most clearly to the transcript-aware judge, but it cannot
   predict answer quality by itself. The judge does not see the retrieved span
   and therefore does not measure faithfulness to the generator's evidence.
2. Word-level recall understates useful evidence coverage when gold spans are
   long or redundant and becomes invalid when a span is misannotated.
3. High recall with low precision can still produce a distracted or incorrect
   answer.
4. ROUGE rewards lexical overlap and BERTScore rewards semantic resemblance;
   neither checks whether claims are supported by the retrieved transcript.
5. Oracle answers are model outputs, not reference answers or guaranteed upper
   bounds. They can omit details, and they inherit gold-span errors.
6. Report BERTScore with enough decimals or paired deltas; values that look
   identical at three decimals are not identical in the artifacts.
7. For the thesis, use retrieval precision/recall, reference-overlap metrics,
   transcript-aware judging, and a clearly documented manual review as
   complementary evidence.

Useful instructor decisions are whether to add a budget-matched oracle,
whether to audit all annotation anomalies under a fixed protocol, and whether
the final claims should emphasize the judge/manual findings over the compressed
ROUGE and BERTScore differences.
