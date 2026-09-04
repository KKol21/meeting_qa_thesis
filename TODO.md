# Thesis experiment TODOs

## Decisions to confirm

- [ ] Confirm whether retrieval evidence should remain budgeted in whitespace-delimited words or use Qwen tokens.
- [ ] Ask whether a tokenizer-based baseline is still needed beyond the implemented 256-word hard-packed baseline.
- [ ] Use `val` for development; decide whether the frozen final evaluation should run once on `test`.
- [ ] Decide whether ELITR Bench remains in scope. The current experiment evaluates QMSum only.

## Code TODOs

### P0 — protect the validity of the current experiment

- [x] Let Wormulon job `4937` finish and download its outputs before changing existing result formats.
- [x] Remove the repeatedly inspected meeting `Bed002` from full aggregate results.
- [x] Replace it with `education_18`, the next meeting in the seed-42 ordering.
- [ ] Change the answer judge so gold transcript evidence is the primary factual source and the reference answer is explicitly non-exhaustive.

### P1 — add a genuine fixed-word baseline

- [x] Add a chunk representation that retains partial-turn text, original turn ID, and word offsets.
- [x] Implement sequential fixed-word chunking that may split speaker turns and has no overlap.
- [x] Exclude repeated speaker/turn labels from the 256-content-word limit and document the choice.
- [x] Ensure word-packed chunks reconstruct source words exactly once, without gaps or duplication.
- [x] Keep the 256-word turn-preserving chunker as a separate soft-limit baseline.
- [x] Rename chunkers unambiguously: `word_packed`, `turn_packed`, and `lumber`.
- [x] Extend retrieval to 18 conditions: 3 chunkers × 3 retrievers × 2 evidence budgets.
- [x] Preserve and reconstruct split-turn fragments without deduplicating by turn ID.
- [x] Score partial-turn evidence using its source turn and selected source-word count.
- [x] Render split turns with repeated speaker attribution and chronological evidence order.
- [x] Add unit tests for long-turn splitting, offsets, empty turns, evidence reconstruction, and ordering.
- [x] Run a one-meeting smoke test before launching the additional conditions.

### P1 — complete quantitative analysis

- [ ] Add retrieval F1 per question and macro-average F1 per condition.
- [ ] Add zero-hit rate per condition.
- [x] Add per-meeting metric tables.
- [x] Add paired chunker differences while holding retriever and evidence budget constant.
- [ ] Add meeting-clustered bootstrap confidence intervals for paired differences.
- [ ] Fix the bootstrap seed and record the number of resamples and confidence level.
- [ ] Report the number of meetings and questions contributing to every aggregate.

### P1 — validate automatic answer evaluation

- [ ] Define a fixed manual-review sample before examining the full results.
- [ ] Include random questions, zero-hit cases, large baseline-versus-Lumber differences, and metric-disagreement cases.
- [ ] Record reference adequacy, evidence sufficiency, answer correctness, and unsupported claims during manual review.
- [ ] Discuss general LLM-judge bias; the Llama 3.3 judge and Qwen2.5 answer models are from different model families.
- [ ] Treat ROUGE and BERTScore as reference-overlap measures rather than direct factual-correctness measures.

### P2 — reporting and reproducibility

- [ ] Extend `report_ablations.py` with F1, zero-hit rate, and confidence intervals. Paired differences are implemented.
- [ ] Mark development meetings separately in reports and exclude them from final aggregates by default.
- [ ] Record dataset split, meeting-selection rule, seed, exact meeting IDs, model revisions, prompts, and decoding settings in the final summary.
- [ ] Record hardware and pinned software versions.
- [ ] Add a compact machine-readable final results table suitable for importing into LaTeX.

## Thesis text TODOs

Use this as a self-contained revision brief for the first peer-review draft. Do not
claim planned analyses as completed; leave clearly marked placeholders where final
results or instructor decisions are still pending.

### 1. Align the experimental-design overview

- [ ] Describe a factorial experiment with three chunkers (`word_packed`, `turn_packed`, and `lumber`), three retrievers, and two evidence budgets, for 18 conditions.
- [ ] Explain the chunkers in one sentence each: word-packed chunks split at 256 content words and repeat attribution; turn-packed chunks greedily pack complete turns under a soft 256-word limit; Lumber places semantic boundaries only between turns. Neither non-semantic baseline overlaps.
- [ ] Describe the separate oracle experiment, which supplies annotated gold evidence directly and compares Qwen2.5 7B, 14B, and quantized 32B answer models.

### 2. Correct the dataset description

- [ ] State that development uses 20 QMSum validation meetings (142 questions) in a seed-42 order, excludes the repeatedly inspected `Bed002`, and replaces it with the next meeting, `education_18`.
- [ ] Say that transcripts are represented as ordered speaker turns with stable source positions, allowing every chunk and retrieved span to be mapped back to the transcript.
- [ ] State clearly whether ELITR Bench is included. If QMSum is the only evaluated dataset, call it "the evaluation dataset," not "the primary dataset."

### 3. Replace the segmentation-method details

- [ ] For word-packed chunking, report the 256-content-word hard limit, repeated speaker labels outside that budget, and preserved source word offsets.
- [ ] For turn-packed chunking, report the implemented 256-whitespace-word limit and preservation of complete turns.
- [ ] For Lumber, state that turns replace the paragraphs in the original method; the original prompt is used on rolling windows of approximately 550 tokens, estimated as `1.2 x whitespace word count`. The boundary model is Qwen2.5-7B-Instruct with greedy decoding, at most 32 new tokens, and seed 42. An invalid boundary triggers one constrained retry.

### 4. Correct retrieval and budget accounting

- [ ] Describe dense retrieval using normalized `Alibaba-NLP/gte-modernbert-base` embeddings ranked by dot product, BM25 with `k_1=1.5` and `b=0.75`, and hybrid retrieval using reciprocal-rank fusion with `k=60`.
- [ ] Call 512 and 1,024 "word budgets" unless the implementation is changed after instructor feedback. Evidence selection uses unique source words and clips the final retrieved unit to the remaining budget.
- [ ] Define gold evidence using QMSum's annotated turn ranges. Define precision, recall, F1, reciprocal rank of the first chunk overlapping gold evidence, and zero-hit rate; report macro-averages across questions.

### 5. Update downstream QA and answer evaluation

- [ ] State that Qwen2.5-14B-Instruct answers every retrieval condition using the same prompt, temperature 0, seed 42, and at most 512 generated tokens.
- [ ] Report ROUGE-1/2/L F1, BERTScore with RoBERTa-large, and an LLM judge score of 1 (incorrect), 2 (partially correct), or 3 (correct). The judge receives the question, candidate answer, QMSum reference, and gold transcript evidence.
- [ ] Explain that QMSum references can be selective or incomplete, so ROUGE and BERTScore measure reference similarity rather than factual correctness. Treat transcript-grounded judging and predefined manual review as complementary evidence, and discuss general LLM-judge bias while noting that the Llama judge and Qwen answer models are from different families.

### 6. State the analysis and reproducibility plan cautiously

- [ ] Compare chunkers pairwise while holding retriever and budget fixed. Report per-meeting results and meeting-clustered bootstrap confidence intervals only once implemented; add the final resample count, confidence level, and seed.
- [ ] Predefine manual-review cases: a random sample, zero-hit cases, large chunker differences, and disagreements between reference metrics and the judge. Distinguish reference-quality, retrieval, and generation failures.
- [ ] Add one compact reproducibility table containing dataset selection, meeting IDs, prompts, model/checkpoint revisions, chunk and evidence budgets, decoding settings, software versions, and hardware. State that per-question segmentations, rankings, evidence, answers, and judgments are retained.

## Suggested order of work

1. Recover and preserve job `4937` results.
2. Freeze the 20-meeting development set without `Bed002`. (Completed.)
3. Smoke-test the new word-packed chunks and provenance chain. (Completed.)
4. Run only the additional or invalidated conditions.
5. Revise and rerun the evidence-first judge.
6. Compute paired statistics and confidence intervals from saved outputs.
7. Freeze the experiment configuration.
8. Rewrite the Method chapter to describe the frozen implementation exactly.
9. Write Results from generated tables and predefined manual-review samples.
