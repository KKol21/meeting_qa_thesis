# Meeting QA chunking

Minimal experiments comparing fixed-token, turn-packed, and LumberChunker
segmentation on QMSum and ELITR-Bench.

The comparison changes only segmentation. All methods share the same canonical
speaker-labelled transcript, dense retriever, exact unique-source-token evidence
budget, answer model, decoding settings, and answer prompt.

Important invariants:

- LumberChunker receives only a sequential rolling window of speaker turns. It
  never receives a whole transcript, question, reference answer, or gold span.
- Its instruction block is the original Table 5 prompt from the
  [LumberChunker paper](https://aclanthology.org/2024.findings-emnlp.377.pdf).
  The paper used Gemini 1.0-Pro; this baseline keeps the algorithm and prompt but
  makes the boundary model an explicit, reportable configuration substitution.
- QMSum and ELITR use the same answer builder in `generation.py`. ELITR's separate
  rubric prompt is used only after generation for evaluation.
- Retrieval is projected to an exact budget of unique source tokens. Overlap is
  deduplicated and the final ranked unit is clipped when necessary.
- The configured [GTE ModernBERT model](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)
  accepts up to 8,192 tokens and is pinned to a repository revision. The
  implementation raises an error instead of silently truncating a chunk.

## Setup

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[experiment]"
```

Place the official data in these ignored directories:

```text
data/raw/qmsum/data/ALL/{val,test}/
data/raw/elitr-bench/data/elitr-bench-qa_{dev,test2}.json
data/raw/elitr-bench/elitr-minuting-corpus-en/{dev,test2}/
```

Then validate both corpora and the baseline configuration:

```powershell
.venv\Scripts\meeting-qa --config configs/baseline.toml validate-config
.venv\Scripts\meeting-qa validate-data
```

Model-backed stages use the OpenAI-compatible endpoint and model IDs in
`configs/baseline.toml`. The key is read only from the configured environment
variable:

```powershell
$env:OPENAI_API_KEY = "your-key"
```

## Incremental smoke run

First inspect the planned calls without loading models or writing artifacts:

```powershell
.venv\Scripts\meeting-qa --config configs/baseline.toml run --dataset qmsum --split val --meeting-id TS3010a --limit-queries 2 --run-dir runs/qmsum-smoke --dry-run
```

Run one stage at a time with the same selection and run directory:

```powershell
.venv\Scripts\meeting-qa --config configs/baseline.toml run --dataset qmsum --split val --stage segment --meeting-id TS3010a --limit-queries 2 --run-dir runs/qmsum-smoke
.venv\Scripts\meeting-qa --config configs/baseline.toml run --dataset qmsum --split val --stage retrieve --meeting-id TS3010a --limit-queries 2 --run-dir runs/qmsum-smoke
.venv\Scripts\meeting-qa --config configs/baseline.toml run --dataset qmsum --split val --stage answer --meeting-id TS3010a --limit-queries 2 --run-dir runs/qmsum-smoke
.venv\Scripts\meeting-qa --config configs/baseline.toml run --dataset qmsum --split val --stage evaluate --meeting-id TS3010a --limit-queries 2 --run-dir runs/qmsum-smoke
```

Each stage is resumable. Successful model calls, embeddings, and segmentations
are cached; malformed Lumber/ELITR outputs are not cached. A shared API-call cap
defaults to 20 uncached calls per invocation. Increase it explicitly for a full
run. Changing code, configuration, models, prompts, data, or selection requires a
new run directory, preventing stale-cache reuse.

For an ELITR smoke run, use
`--dataset elitr --split dev --meeting-id meeting_en_dev_001`. ELITR defaults to
the 512-token budget; QMSum defaults to
256, 512, and 1,024 for retrieval and generates only at 512.

## Final experiment

Report QMSum `test` and ELITR `test2`; use development examples only as smoke
tests. The three methods are enabled by default.

```powershell
.venv\Scripts\meeting-qa --config configs/baseline.toml run --dataset qmsum --split test --stage all --run-dir runs/qmsum-test --max-api-calls 5000
.venv\Scripts\meeting-qa --config configs/baseline.toml run --dataset elitr --split test2 --stage all --run-dir runs/elitr-test2 --max-api-calls 5000
```

QMSum records token precision/recall/F1 and zero-hit at every budget, plus
ROUGE-1/2/L F1 at 512. ELITR records the official 1–10 reference-answer judge
score and question-type/answer-position metadata; it has no gold evidence spans,
so no retrieval-accuracy score is invented. Do not combine the two datasets into
one score.

Each run writes a fingerprinted `manifest.json`, auditable caches, and stage
artifacts such as `segments.jsonl`, `retrieval.jsonl`, `answers.jsonl`,
`judgments.jsonl`, `metrics.jsonl`, and an uncached-call `calls.jsonl` containing
the provider-returned model and system fingerprint when available. After
evaluation, aggregate macro scores,
diagnostics, and paired meeting-cluster bootstrap intervals with:

```powershell
.venv\Scripts\meeting-qa summarize --run-dir runs/qmsum-test
.venv\Scripts\meeting-qa summarize --run-dir runs/elitr-test2
```

## Development checks

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
```
