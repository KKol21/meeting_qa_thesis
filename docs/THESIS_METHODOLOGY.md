# Thesis methodology summary

This document contains the methodological choices that should be stated and
defended in the thesis. Implementation and cluster details belong in the
reproducibility appendix rather than the main method chapter.

## Research question

The experiment tests whether LLM-selected semantic boundaries improve
retrieval and downstream question answering over simpler meeting-transcript
segmentations. The central comparison is not Lumber against isolated speaker
turns, because those chunks are much smaller. The stronger comparisons are
Lumber against two deterministic, size-controlled baselines under identical
retrieval and evidence-budget conditions.

The primary claim should concern evidence retrieval. Answer-generation metrics
are downstream corroboration because they also depend on the answer model,
reference-answer quality, and automatic evaluator.

## Data and split discipline

Only QMSum specific-query questions are used. Each example contains a question,
a reference answer, and one or more annotated inclusive turn ranges. Parameter
selection and sensitivity analyses use a fixed 20-meeting validation subset
with 142 questions and are therefore exploratory rather than independent
confirmation. After freezing the design, the final experiment uses all 35
QMSum test meetings and their 244 specific-query questions.

## Chunking conditions

Validation compared four representations of every transcript:

1. **Single turn:** every speaker turn is a separate chunk. This is a
   deliberately fine-grained baseline, not a size-matched baseline.
2. **Turn packed:** consecutive complete turns are greedily packed to a
   256-word soft limit. A turn longer than 256 words remains intact and can
   exceed the limit.
3. **Word packed:** consecutive transcript words are packed to a strict
   256-word limit. Long turns may be divided, with the speaker label repeated
   for each continuation.
4. **Lumber:** a Qwen2.5-14B instruction model receives a local sequence of
   numbered turns and identifies the first clear content shift. Segmentation is
   sequential and boundaries remain legal turn boundaries. Greedy decoding is
   used. The selected window target is 1,000 pseudo-tokens.

The held-out test grid omits the single-turn condition. It is not size matched,
and validation already established that comparisons against it mainly measure
the benefit of aggregating turns. The final confirmatory comparisons are
Lumber versus turn-packed and word-packed chunks.

Lumber's `target_tokens` value is not a tokenizer count. It follows the adapted
Lumber procedure's deterministic approximation over rendered text. Chunk and
evidence budgets use whitespace-separated transcript-content words, excluding
speaker labels and turn identifiers. These units must not be presented as
interchangeable.

Turn identifiers are retained for reconstruction and scoring but excluded from
retrieval representations. Speaker labels are retained because participant
identity can be relevant to QMSum questions.

## Parameter selection

Parameters were chosen before the final evaluation using descriptive corpus
statistics and validation-only sensitivity analyses:

- 256 words accommodates more than 99.6% of QMSum turns without splitting and
  creates approximately 14-turn packed chunks on average.
- Lumber targets from 500 to 1,500 were compared. The 1,000 target was retained
  because its median chunk size closely matches the deterministic baselines and
  its retrieval F1 was competitive, limiting geometric confounding.
- A 1,024-word evidence budget is the primary operating point. The 512- and
  2,048-word conditions test sensitivity to constrained and expanded context.

The target sweep, deterministic size sweep, clipping analysis, shuffled-boundary
control, and boundary-model check are secondary analyses. They must not be
presented as additional independent test sets.

## Retrieval and evidence selection

Retrieval is performed within the transcript associated with each question.
Every chunker is evaluated with the same three retrieval methods:

- normalized dense embeddings from GTE-ModernBERT-base with cosine similarity;
- Okapi BM25 (`k1 = 1.5`, `b = 0.75`);
- reciprocal-rank fusion of the dense and BM25 rankings (`k = 60`).

This produces nine prespecified environments per chunker: three retrievers
crossed with evidence budgets of 512, 1,024, and 2,048 words. Chunks are selected
in retrieval order until the budget is exhausted. The last selected chunk is
clipped to enforce the exact word budget, after which selected excerpts are
presented to the answer model in chronological transcript order. The same
selection and rendering procedure is used for every chunker.

## Retrieval outcomes

QMSum evidence annotations are turn-level. All words belonging to an annotated
turn are consequently treated as relevant. The principal metric is evidence
recall:

> relevant retrieved words / words in the union of gold evidence turns.

Precision, F1, and the proportion of questions with zero retrieved gold words
are reported alongside recall. First-overlap reciprocal rank is secondary
because the probability that a chunk overlaps an annotated turn depends on
chunk size.

This scoring convention is transparent but imperfect: it treats every word in
a gold turn as relevant even when only part of that turn supports the answer.

## Answer generation and evaluation

For the end-to-end comparison, Qwen2.5-14B answers every question from only the
selected evidence. The answer prompt and decoding settings are identical across
retrieval conditions. Oracle runs replace retrieved evidence with the annotated
gold turns and compare 7B, 14B, and quantized 32B answer models. Oracle results
estimate the answer model's performance when retrieval error is removed; they
are not an attainable retrieval condition.

Generated answers are assessed using ROUGE-1/2/L, rescaled BERTScore, and a
separate quantized Llama-3.3-70B judge. The judge receives the question,
reference answer, gold transcript evidence, and candidate answer, and assigns:

- 1: incorrect, invalid, or materially unsupported;
- 2: supported but incomplete, or containing a minor unsupported claim;
- 3: correct, sufficiently complete, and grounded.

Giving the judge both the reference and gold evidence is defensible because the
reference supplies the expected answer content while the transcript resolves
omissions or imperfections in that reference. Judge scores remain model-based
measurements rather than human ground truth and should be interpreted together
with retrieval metrics and qualitative inspection.

## Aggregation and uncertainty

Questions are first averaged within each meeting, after which meetings receive
equal weight. This avoids allowing meetings with more annotated questions to
dominate the result.

The headline comparison averages over the nine prespecified retrieval
environments. For meeting *m* and baseline *b*, it is

```text
d(m,b) = mean over environments e [recall(m,Lumber,e) - recall(m,b,e)].
```

The reported effect is the equal-weight mean of `d(m,b)` across meetings.
Uncertainty is estimated using 10,000 non-parametric bootstrap samples of whole
meetings with replacement. The 2.5th and 97.5th percentiles form a paired 95%
confidence interval. This preserves dependence among questions and among the
nine conditions from the same meeting.

Validation effects include the exploratory single-turn contrast. Held-out test
effects are reported against turn-packed and word-packed baselines. The nine
cell-specific paired differences show whether
the result varies with retriever or evidence budget. A significant result in
one cell and a non-significant result in another is not, by itself, evidence
that the two cell effects differ; such claims require a direct interaction
contrast. Cell-level analyses are exploratory and are not corrected for
multiple comparisons.

## Robustness analyses

The following checks isolate plausible alternative explanations:

- **Chunk-size sweep:** repeats deterministic baselines at 128, 256, and 512
  words.
- **Evidence clipping:** compares exact clipping with dropping or fully
  including the final chunk.
- **Boundary-shuffled control:** preserves legal turn boundaries, chunk count,
  and approximately the Lumber chunk-size distribution while destroying the
  semantic locations of boundaries.
- **Boundary-model check:** compares 7B, 14B, and quantized 32B Qwen models on
  five fixed validation meetings while holding all other settings constant.

The model check is only a screening analysis: five meeting clusters can reveal
a large failure or improvement but cannot establish equivalence. Any model
change suggested by it should be confirmed on the complete validation subset
before altering a frozen final configuration.

## Appropriate strength of conclusions

The design supports statements about associations between chunking strategy,
retrieval coverage, and answer quality on QMSum specific-query QA. It does not
support a universal claim that semantic chunking is or is not useful for all
RAG tasks. In particular, LumberChunker was introduced for long-form narrative
documents, whereas meetings contain interruptions, short backchannels, topic
recurrence, and evidence distributed across speakers.

The most defensible conclusion should distinguish geometry from semantics:

> Lumber should first be compared with size-controlled deterministic chunks.
> An advantage over single turns alone demonstrates the value of aggregating
> turns, not necessarily the value of semantic boundary selection. Evidence
> that boundary placement itself matters comes from paired comparisons with
> packed baselines and the boundary-shuffled control.

## Citations that should appear in the chapter

The final thesis text should cite the original sources for QMSum,
LumberChunker, BM25, reciprocal-rank fusion, ROUGE, BERTScore, and the embedding
and generation models. Package documentation and Slurm commands are
reproducibility material, not substitutes for methodological citations.
