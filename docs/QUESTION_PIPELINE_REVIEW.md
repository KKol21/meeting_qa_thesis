# One QMSum question through the pipeline

This is a code-review map of the methodology, not the cluster workflow. It
follows one specific QMSum question through every experimental condition and
names the code and saved fields that determine the result.

The run is a QMSum validation-set development experiment. Its semantic chunker
is a LumberChunker adaptation, not a model- and dataset-exact reproduction.

```text
QMSum JSON
  -> Meeting + Question
  -> three chunk views
  -> dense, BM25, and hybrid rankings
  -> 512- and 1024-word evidence selections
  -> retrieval metrics
  -> Qwen2.5-14B answers
  -> ROUGE + BERTScore + Llama judge
  -> question, meeting, and paired meeting aggregates

In parallel: annotated gold turns -> 7B/14B/32B oracle answers -> same metrics
```

The full preset fixes the three chunkers, three retrievers, two evidence
budgets, generation settings, and evaluation models in
[`ablation-full.toml:10-61`](../src/configs/ablation-full.toml#L10-L61).
Consequently, one question produces 18 retrieved-evidence answers. The
`oracle-14b` result is the fairest answer-model control because it uses the
same Qwen2.5-14B model and prompt; only its evidence source differs.

## 1. Load the meeting and question

[`load_meeting()`](../src/meeting_qa_chunking/qmsum.py#L29-L48) maps:

- `meeting_transcripts` to `Turn(id, speaker, text)`, assigning zero-based,
  contiguous turn IDs;
- `specific_query_list` to
  `Question(text, reference_answer, relevant_turn_ranges)`;
- `relevant_text_span` to inclusive `(start_turn, end_turn)` ranges.

These ranges are both the oracle evidence and the retrieval relevance labels.
A bad range therefore affects recall, oracle answering, and the LLM judge's
gold context.

## 2. Create the three chunk views

[`build_chunk_sets()`](../src/meeting_qa_chunking/evidence_preparation.py#L18-L45)
constructs three non-overlapping views of the same ordered turns.

### Turn-packed baseline

[`chunk_turn_packed()`](../src/meeting_qa_chunking/chunking.py#L79-L100)
greedily adds complete turns until the next turn would exceed 256 content
words. It is a soft limit: a single turn longer than 256 words remains intact.

### Word-packed baseline

[`chunk_word_packed()`](../src/meeting_qa_chunking/chunking.py#L103-L145)
fills hard 256-content-word chunks and splits turns when required. A split
fragment retains `turn_id`, `speaker`, and `start_word`. Rendering repeats the
speaker attribution on every fragment
([`Chunk.text`](../src/meeting_qa_chunking/chunking.py#L61-L71)). Speaker labels
and IDs enter the retrieval text but are not charged to the content-word
budget.

### Lumber semantic chunks

Lumber segmentation is performed once per meeting, before questions are
ranked. [`build_window()`](../src/meeting_qa_chunking/lumber_prompt.py#L29-L47)
starts at the next unprocessed turn and adds complete turns until the original
Lumber estimate, `round(1.2 * whitespace words)`, exceeds 550 tokens
([`estimate_tokens()`](../src/meeting_qa_chunking/lumber_prompt.py#L23-L26)).
The full transcript is never sent in one prompt.

[`lumber_chunks()`](../src/meeting_qa_chunking/lumber.py#L43-L82) asks
Qwen2.5-7B for the first turn that begins a new topic. The returned boundary
starts the next chunk; the preceding turns close the current chunk. One
constrained retry follows an invalid response. During retrieval,
[`load_lumber_chunks()`](../src/meeting_qa_chunking/lumber.py#L23-L40)
reconstructs the complete-turn chunks and verifies exact transcript coverage.

The segmentation artifact is written at
[`ablation_segment.py:98-125`](../src/stages/ablation_segment.py#L98-L125) and
saves:

```text
meeting_id, turn_count, model_calls, cache_hits, provenance
chunks[]: index, start_turn, end_turn, word_count
decisions[]: boundary_turn, raw_response, cache_hit
```

## 3. Rank each chunk view for the question

The question loop and three ranking branches are in
[`ablation_retrieval.py:210-240`](../src/stages/ablation_retrieval.py#L210-L240).

- [`rank_chunks()`](../src/meeting_qa_chunking/retrieval.py#L80-L103)
  encodes the question and rendered chunks with GTE-ModernBERT. Normalized
  embeddings make their dot product cosine similarity.
- [`rank_chunks_bm25()`](../src/meeting_qa_chunking/retrieval.py#L106-L142)
  uses lowercase `\w+` tokens and Okapi BM25 with `k1=1.5`, `b=0.75`.
- [`reciprocal_rank_fusion()`](../src/meeting_qa_chunking/retrieval.py#L145-L155)
  combines dense and BM25 ranks as `1 / (60 + rank)` without mixing their raw
  scores.

All ties are resolved by the earlier chunk index. A condition name is
`chunker__retriever__w<budget>`, defined by
[`ConditionSpec.name`](../src/meeting_qa_chunking/config.py#L62-L77).

## 4. Select and score evidence

For each ranking, [`select_evidence()`](../src/meeting_qa_chunking/evidence.py#L119-L164)
visits chunks in rank order until it has 512 or 1024 content words. It:

- clips the final `ChunkPart` when the remaining budget is smaller;
- deduplicates words by `(turn_id, absolute_word_offset)`;
- records the contributing chunk indices in retrieval order.

[`score_evidence()`](../src/meeting_qa_chunking/evidence.py#L167-L193) labels a
retrieved word relevant when its source turn lies in any annotated gold range:

```text
precision = relevant retrieved words / retrieved words
recall    = relevant retrieved words / all words in annotated gold turns
```

This is word coverage of gold-labelled turns, not answer-fact recall. The
first-overlap rank is separately calculated at chunk level by
[`first_relevant_chunk_rank()`](../src/meeting_qa_chunking/evidence.py#L102-L116).

The per-question retrieval record is assembled at
[`ablation_retrieval.py:241-260`](../src/stages/ablation_retrieval.py#L241-L260):

```text
question_index, question, reference_answer
results[condition]:
  precision, recall
  first_overlap_rank, first_overlap_reciprocal_rank
  retrieved_words, relevant_retrieved_words, gold_words
  selected_chunk_indices
```

The artifact does not retain the complete ranking or raw retrieval scores.

## 5. Prepare evidence for answering

Retrieved evidence is rebuilt from `selected_chunk_indices` by
[`prepare_retrieved_evidence()`](../src/meeting_qa_chunking/evidence_preparation.py#L67-L119).
[`reconstruct_evidence()`](../src/meeting_qa_chunking/evidence.py#L52-L66)
reruns the same budget logic and verifies `retrieved_words`. Selection remains
rank-based, but [`render_evidence()`](../src/meeting_qa_chunking/evidence.py#L34-L49)
sorts the selected fragments into chronological transcript order before the
answer model sees them.

Oracle evidence is different: [`prepare_oracle_evidence()`](../src/meeting_qa_chunking/evidence_preparation.py#L48-L64)
renders every annotated gold turn in transcript order without a 512- or
1024-word cap. It is therefore an answer-model control, not a budget-matched
retrieval condition.

## 6. Generate and initially score the answer

[`build_answer_prompt()`](../src/meeting_qa_chunking/answering.py#L10-L15)
combines the shared instruction, question, and evidence. The answer model does
not see the reference answer. The instruction explicitly asks for every
supported relevant detail
([`answer.txt:1-6`](../src/meeting_qa_chunking/prompts/answer.txt#L1-L6)).

The retrieval stage and `oracle-14b` both use Qwen2.5-14B with greedy
generation and at most 512 new tokens. The per-condition call and record are
at [`ablation_answer.py:219-247`](../src/stages/ablation_answer.py#L219-L247).
[`score_answer()`](../src/meeting_qa_chunking/answering.py#L18-L24) immediately
computes stemmed ROUGE-1, ROUGE-2, and ROUGE-L F1 against the QMSum reference.

Each answer result saves:

```text
answer, rouge_f1, cache_hit, evidence_words

retrieved source additionally:
  evidence_order, selected_chunk_indices
  retrieval_precision, retrieval_recall

oracle source additionally:
  gold_turn_ranges
```

The meeting-level answer artifact also saves `source`, `conditions`,
`answer_model`, `model_calls`, `cache_hits`, `questions`, and `provenance`
([`ablation_answer.py:250-263`](../src/stages/ablation_answer.py#L250-L263)).

## 7. Evaluate the answer

[`load_stage()`](../src/stages/ablation_evaluate.py#L52-L77) joins each saved
candidate with the question, reference, and complete annotated gold evidence.

- [`add_bertscore()`](../src/stages/ablation_evaluate.py#L80-L93) compares only
  candidate and reference with RoBERTa-large. It does not inspect retrieved or
  gold evidence. Baseline rescaling is disabled
  ([`ablation_evaluate.py:249-259`](../src/stages/ablation_evaluate.py#L249-L259)).
- [`build_judge_prompt()`](../src/meeting_qa_chunking/judging.py#L12-L24) gives
  Llama-3.3-70B the question, reference, annotated gold evidence, and
  candidate, but not the retrieved evidence used to generate that candidate.
  It therefore checks transcript-aware answer quality rather than evidence
  faithfulness. The rubric maps incorrect, partial, and correct answers to 1,
  2, and 3 ([`judge.txt:1-12`](../src/meeting_qa_chunking/prompts/judge.txt#L1-L12)).

The evaluation loop saves one record per question and condition at
[`ablation_evaluate.py:280-303`](../src/stages/ablation_evaluate.py#L280-L303):

```text
meeting_id, question_index, condition
bertscore: precision, recall, f1
judge: score, reason, raw_response, cache_hit
```

## 8. Aggregate

Retrieval aggregation is implemented in
[`ablation_retrieval.py:46-120`](../src/stages/ablation_retrieval.py#L46-L120),
ROUGE aggregation in
[`ablation_answer.py:33-109`](../src/stages/ablation_answer.py#L33-L109), and
BERTScore/judge aggregation in
[`ablation_evaluate.py:96-184`](../src/stages/ablation_evaluate.py#L96-L184).
They save:

- `question_average`: all questions pooled, weighting meetings by their
  question count;
- `per_meeting`: questions first averaged within each meeting;
- `meeting_average`: equal-weight mean of meeting means;
- `paired_meeting_average`: Lumber-minus-baseline differences calculated
  within meetings before averaging.

The report uses `meeting_average` for primary tables. Judge distributions are
raw question counts. The top-level summary only collects and hashes the stage
summaries ([`summarize_ablations.py:20-47`](../src/tools/summarize_ablations.py#L20-L47)).

## Methodological points to defend

1. Budgets use whitespace-delimited content words, while Lumber's window uses
   a `1.2 * words` token approximation.
2. Turn-packed is soft-sized; word-packed is hard-sized; Lumber preserves
   complete turns but has unconstrained semantic chunk sizes.
3. Speaker labels and turn IDs affect retrieval representations but do not
   consume the content-word budget.
4. Evidence membership is chosen in retrieval order but presented
   chronologically.
5. Recall is word coverage of annotated turns, not coverage of atomic answer
   facts.
6. Oracle evidence is complete and uncapped, so it is not an equal-budget
   comparison.
7. ROUGE and BERTScore see only the reference; the judge additionally sees
   annotated evidence.
8. Meeting-macro and paired within-meeting results are the appropriate primary
   comparisons; pooled question means are descriptive.
9. Chunking strategy is confounded with granularity in this run: mean chunk
   sizes are 167 words for Lumber, 226 for turn-packed, and 252 for
   word-packed. Results compare complete configurations, not semantic boundary
   quality alone.
