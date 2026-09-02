# Thesis experiment TODOs

## Decisions to confirm

- [ ] Confirm whether retrieval evidence should remain budgeted in whitespace-delimited words or use Qwen tokens.
- [ ] Choose the fixed-token chunk size. It should be approximately comparable to the 256-word turn-packed baseline.
- [ ] Decide whether the final evaluation is an untouched validation subset or a test-set experiment.
- [ ] Decide whether ELITR Bench remains in scope. The current experiment evaluates QMSum only.

## Code TODOs

### P0 — protect the validity of the current experiment

- [x] Let Wormulon job `4937` finish and download its outputs before changing existing result formats.
- [ ] Remove the repeatedly inspected meeting `Bed002` from final aggregate results.
- [ ] Either report 19 untouched meetings or run one untouched replacement meeting to retain 20.
- [ ] Change the answer judge so gold transcript evidence is the primary factual source and the reference answer is explicitly non-exhaustive.
- [ ] Increment the evaluation version and rerun evaluation only; reuse cached segmentations, retrieval results, and answers.

### P1 — add a genuine fixed-token baseline

- [ ] Add a chunk representation that can retain partial-turn text together with its original turn ID and source offsets.
- [ ] Implement sequential fixed-token chunking that may split speaker turns and has no overlap.
- [ ] Pin and record the tokenizer name and revision used for chunk boundaries.
- [ ] Define whether speaker labels count toward the token limit and apply that choice consistently.
- [ ] Ensure fixed-token chunks reconstruct the complete transcript exactly once, without gaps or duplicated source text.
- [ ] Keep the existing 256-word turn-preserving chunker as a separate baseline.
- [ ] Rename chunkers unambiguously, for example `fixed_token`, `turn_packed`, and `lumber`.
- [ ] Extend retrieval configurations from 12 to 18 conditions: 3 chunkers × 3 retrievers × 2 evidence budgets.
- [ ] Update evidence selection and reconstruction so partial turns remain traceable and are not incorrectly deduplicated by turn ID.
- [ ] Update retrieval scoring so partial-turn evidence receives the correct source-word overlap credit.
- [ ] Update answer generation and manual-review reports to render partial turns with their speaker and source position.
- [ ] Preserve caches for the existing conditions so only the new fixed-token conditions require generation.
- [ ] Add unit tests for exact transcript coverage, long-turn splitting, chunk limits, source mapping, evidence reconstruction, and deterministic boundaries.
- [ ] Run a one-meeting smoke test before launching the additional conditions.

### P1 — complete quantitative analysis

- [ ] Add retrieval F1 per question and macro-average F1 per condition.
- [ ] Add zero-hit rate per condition.
- [ ] Add per-meeting metric tables.
- [ ] Add paired chunker differences while holding retriever and evidence budget constant.
- [ ] Add meeting-clustered bootstrap confidence intervals for paired differences.
- [ ] Fix the bootstrap seed and record the number of resamples and confidence level.
- [ ] Report the number of meetings and questions contributing to every aggregate.

### P1 — validate automatic answer evaluation

- [ ] Define a fixed manual-review sample before examining the full results.
- [ ] Include random questions, zero-hit cases, large fixed-versus-Lumber differences, and metric-disagreement cases.
- [ ] Record reference adequacy, evidence sufficiency, answer correctness, and unsupported claims during manual review.
- [ ] Report possible same-family bias because the judge and answer models are Qwen models.
- [ ] Treat ROUGE and BERTScore as reference-overlap measures rather than direct factual-correctness measures.

### P2 — reporting and reproducibility

- [ ] Extend `report_ablations.py` to include F1, zero-hit rate, confidence intervals, and paired differences.
- [ ] Mark development meetings separately in reports and exclude them from final aggregates by default.
- [ ] Record dataset split, meeting-selection rule, seed, exact meeting IDs, model revisions, prompts, and decoding settings in the final summary.
- [ ] Record hardware and pinned software versions.
- [ ] Add a compact machine-readable final results table suitable for importing into LaTeX.

## Thesis text TODOs

Use this as a self-contained revision brief for the first peer-review draft. Do not
claim planned analyses as completed; leave clearly marked placeholders where final
results or instructor decisions are still pending.

### 1. Align the experimental-design overview

- [ ] Describe a factorial experiment with segmentation as the primary factor. The planned final grid has three chunkers (`fixed_token`, `turn_packed`, and `lumber`), three retrievers (dense, BM25, and hybrid), and two evidence budgets (512 and 1,024 words), for 18 conditions. If fixed-token chunking is not completed, describe only the two implemented chunkers and 12 conditions.
- [ ] Explain the chunkers in one sentence each: fixed-token chunks may split turns; turn-packed chunks greedily pack complete turns up to 256 words; Lumber uses an LLM to place semantic boundaries only between turns. Neither fixed baseline overlaps.
- [ ] Describe the separate oracle experiment, which supplies annotated gold evidence directly and compares Qwen2.5 7B, 14B, and quantized 32B answer models.

### 2. Correct the dataset description

- [ ] State that the experiment uses a fixed subset of QMSum validation meetings selected with seed 42. Add final meeting and question counts after excluding the development meeting `Bed002` or replacing it with an untouched meeting.
- [ ] Say that transcripts are represented as ordered speaker turns with stable source positions, allowing every chunk and retrieved span to be mapped back to the transcript.
- [ ] State clearly whether ELITR Bench is included. If QMSum is the only evaluated dataset, call it "the evaluation dataset," not "the primary dataset."

### 3. Replace the segmentation-method details

- [ ] For fixed-token chunking, report the final tokenizer/checkpoint, token limit, whether speaker labels count toward it, and that source offsets are preserved when a turn is split. Leave these as explicit placeholders until the implementation decision is frozen.
- [ ] For turn-packed chunking, report the implemented 256-whitespace-word limit and preservation of complete turns.
- [ ] For Lumber, state that turns replace the paragraphs in the original method; the original prompt is used on rolling windows of approximately 550 tokens, estimated as `1.2 x whitespace word count`. The boundary model is Qwen2.5-7B-Instruct with temperature 0.1 and seed 42. An invalid boundary triggers one constrained retry. Put the full prompt and exact checkpoint revision in an appendix or configuration table.

### 4. Correct retrieval and budget accounting

- [ ] Describe dense retrieval using normalized `Alibaba-NLP/gte-modernbert-base` embeddings ranked by dot product, BM25 with `k_1=1.5` and `b=0.75`, and hybrid retrieval using reciprocal-rank fusion with `k=60`.
- [ ] Call 512 and 1,024 "word budgets" unless the implementation is changed after instructor feedback. Evidence selection uses unique source words and clips the final retrieved unit to the remaining budget.
- [ ] Define gold evidence using QMSum's annotated turn ranges. Define precision, recall, F1, reciprocal rank of the first chunk overlapping gold evidence, and zero-hit rate; report macro-averages across questions.

### 5. Update downstream QA and answer evaluation

- [ ] State that Qwen2.5-14B-Instruct answers every retrieval condition using the same prompt, temperature 0, seed 42, and at most 512 generated tokens.
- [ ] Report ROUGE-1/2/L F1, BERTScore with RoBERTa-large, and an LLM judge score of 1 (incorrect), 2 (partially correct), or 3 (correct). The judge receives the question, candidate answer, QMSum reference, and gold transcript evidence.
- [ ] Explain that QMSum references can be selective or incomplete, so ROUGE and BERTScore measure reference similarity rather than factual correctness. Treat transcript-grounded judging and predefined manual review as complementary evidence, while acknowledging same-Qwen-family judge bias.

### 6. State the analysis and reproducibility plan cautiously

- [ ] Compare chunkers pairwise while holding retriever and budget fixed. Report per-meeting results and meeting-clustered bootstrap confidence intervals only once implemented; add the final resample count, confidence level, and seed.
- [ ] Predefine manual-review cases: a random sample, zero-hit cases, large chunker differences, and disagreements between reference metrics and the judge. Distinguish reference-quality, retrieval, and generation failures.
- [ ] Add one compact reproducibility table containing dataset selection, meeting IDs, prompts, model/checkpoint revisions, chunk and evidence budgets, decoding settings, software versions, and hardware. State that per-question segmentations, rankings, evidence, answers, and judgments are retained.

## Suggested order of work

1. Recover and preserve job `4937` results.
2. Exclude `Bed002` and settle the final meeting set.
3. Implement and smoke-test fixed-token chunks.
4. Run only the additional or invalidated conditions.
5. Revise and rerun the evidence-first judge.
6. Compute paired statistics and confidence intervals from saved outputs.
7. Freeze the experiment configuration.
8. Rewrite the Method chapter to describe the frozen implementation exactly.
9. Write Results from generated tables and predefined manual-review samples.
