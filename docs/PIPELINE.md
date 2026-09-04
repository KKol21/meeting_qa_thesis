# Experiment pipeline

This is the offline orientation guide for the active QMSum experiment. Start
here when returning to the code after a break. The short version is:

```text
QMSum JSON
    |
    +--> turn-packed chunks -------------+
    +--> word-packed chunks -------------+--> retrieval --> evidence --> answers
    +--> local Lumber segmentation ------+       |                        |
                                                 +--> retrieval metrics   +--> ROUGE
                                                                          +--> BERTScore
                                                                          +--> LLM judge
```

The experiment asks whether semantic chunks improve retrieval and downstream
meeting-question answering compared with two non-semantic baselines.

## Where things live

```text
src/
  stages/                 four ordered experiment stages
  tools/                  reporting and manual-inspection commands
  meeting_qa_chunking/    reusable data, chunking, retrieval, and model code
    prompts/              version-controlled prompts, separate from Python
  configs/                smoke/full presets and the 20-meeting manifest
  wormulon/               Slurm jobs and their Python environment wrapper
docs/vendor/              selected third-party documentation for offline use
data/raw/qmsum/            local QMSum JSON (not uploaded wholesale)
runs/                      generated/fetched artifacts
.cache/                    reusable model responses and embeddings
```

The four files in `src/stages/` are entry points. Most logic that is worth
testing or reusing lives in `src/meeting_qa_chunking/`. The three files in
`src/tools/` consume saved JSON and do not run the experimental models.

The TOML file is the executable experiment definition. It selects meetings,
models, chunkers, retrieval parameters, generation settings, evaluation
settings, and output paths. `meeting_qa_chunking.run_preset` expands it into
separate stage processes. Model tags resolve to pinned repositories and
revisions in `config.py`; Slurm contains no experiment parameters.

## Stage 1: semantic segmentation

Entry point: `src/stages/ablation_segment.py`

1. `qmsum.load_meeting` converts one QMSum JSON file into `Meeting`, `Turn`,
   and `Question` dataclasses. Turn IDs are zero-based positions in the
   transcript.
2. `lumber.lumber_chunks` builds a **local** window beginning at the next
   unprocessed turn. It never sends the whole transcript to the model.
3. `lumber_prompt.build_window` adds complete turns until the window exceeds
   the 550-token target. The token count is the original LumberChunker
   approximation: `round(1.2 * number_of_words)`.
4. The prompt in `prompts/lumberchunker.txt` asks Qwen2.5-7B-Instruct for the
   first turn whose content changes relative to the preceding turns.
5. That boundary closes one chunk. The next window begins at the boundary and
   the process repeats.

An invalid response is retried once with the same task plus an explicit list
of valid IDs. The run-scoped `segmentation/<meeting>.json` stores inclusive
turn ranges and raw model decisions. Boundary generation is greedy with at
most 32 new tokens. Existing files are fingerprinted, validated, and reused,
so rerunning is resumable at meeting level. Model responses are also cached in
`.cache/lumber/` by model, revision, generation settings, and full prompt.

Important distinction: 550 is a local **segmentation-window target**, not the
size of the final semantic chunks and not the retrieval evidence budget.

## Stage 2: chunk retrieval

Entry point: `src/stages/ablation_retrieval.py`

For every meeting it constructs three complete, non-overlapping views:

- `turn_packed`: greedily packs complete turns under a soft 256-word limit;
- `word_packed`: enforces a hard 256-content-word limit, splitting long turns
  and repeating the turn ID and speaker label on each continuation;
- `lumber`: reconstructs the saved semantic turn ranges from stage 1.

Speaker labels and turn IDs are not charged to either content-word limit. The
word-packed baseline stores original word offsets so two fragments from the
same long turn are not accidentally deduplicated.

For each question, every chunk view is ranked three ways:

- `dense`: cosine similarity from normalized
  `Alibaba-NLP/gte-modernbert-base` embeddings;
- `bm25`: the small deterministic Okapi BM25 implementation in
  `retrieval.py` (`k1=1.5`, `b=0.75`);
- `hybrid`: reciprocal-rank fusion of dense and BM25 ranks (`k=60`).

Ranked chunks are selected under 512- and 1024-word budgets. The chosen content
is then rendered in chronological transcript order for dialogue coherence.
Selection may clip the last fragment. This gives 3 chunkers x 3 retrievers x
2 budgets = 18 conditions.

The primary retrieval metrics are evidence precision and recall at equal word
budgets. First-overlap reciprocal rank is retained as a diagnostic but not
used for winner claims because larger chunks are more likely to overlap a gold
turn. Gold relevance is defined by QMSum's annotated inclusive turn ranges.
Consequently, any selected fragment of a gold-labelled turn counts as relevant.
Chunk embeddings live in
`.cache/embeddings/`; the question embedding is recomputed for each ranking
call.

Output: `runs/ablations/<run>/retrieval/<meeting>.json` plus `summary.json`.

## Stage 3: answer generation

Entry point: `src/stages/ablation_answer.py`

There are two evidence sources:

- `oracle`: QMSum's annotated evidence turns. This isolates answer-model
  capacity from retrieval errors.
- `retrieval`: reconstructed stage-2 evidence. This measures the end-to-end
  pipeline.

The Slurm ablation currently runs oracle evidence with Qwen2.5 7B, 14B, and a
prequantized 4-bit 32B checkpoint. Qwen2.5-14B answers all 18 retrieved-evidence
conditions. Every model revision is pinned in `config.py`; the preset chooses
model tags.

`evidence_preparation.py` prepares evidence, `answering.py` combines it with the question,
and `local_model.py` applies the model's chat template and greedy generation.
The actual instruction is in `prompts/answer.txt`. Responses are cached in
`.cache/answers/`. ROUGE-1, ROUGE-2, and ROUGE-L F1 are saved immediately, but
they should not be treated as the only answer-quality measure.

Output: `runs/ablations/<run>/answers/<answer-stage>/<meeting>.json`.

## Stage 4: answer evaluation

Entry point: `src/stages/ablation_evaluate.py`

The evaluator adds two metrics to every saved candidate:

1. BERTScore precision/recall/F1 against the QMSum reference answer, using
   `FacebookAI/roberta-large` at layer 17.
2. A 1--3 judgment from the prequantized 4-bit Llama-3.3-70B-Instruct model:
   1 = invalid/incorrect, 2 = partially correct, 3 = correct.

The judge sees the question, reference answer, gold transcript evidence, and
candidate answer. Its instruction is `prompts/judge.txt`. Because some QMSum
reference answers are incomplete or awkward, the gold evidence is included so
the judge can recognize a supported answer that is phrased differently.

BERTScore is loaded first and then explicitly deleted before the 70B judge is
loaded. This hand-off is why `gc.collect()` and `torch.cuda.empty_cache()` are
present. Judgments are cached in `.cache/judgments/`.

Output: `runs/ablations/<run>/evaluation/<answer-stage>.json`.

## Summaries and manual review

- `src/tools/summarize_ablations.py` collects the small stage summaries into
  the run's top-level `summary.json`.
- `src/tools/report_ablations.py` writes `report.md` plus `review.md`. The
  review file includes every question and condition with retrieved span,
  oracle span, reference, candidate, and metrics.
- `src/tools/inspect_retrieval_failure.py` prints one retrieval case without
  loading an embedding or language model.

Summaries contain both question-weighted averages and meeting-macro averages.
The report uses meeting-macro values and includes paired within-meeting Lumber
differences against both non-semantic baselines.

Generated files are deliberately verbose JSON: they are experimental records,
not application APIs. Reporting retains a small read-only adapter for the
existing version-1 run; new experiment stages write version-2 artifacts.

## Running locally

The repository uses a `src/` layout. In PowerShell, expose that directory once
before running entry-point files directly:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m meeting_qa_chunking.run_preset --preset src/configs/ablation-smoke.toml --dry-run
python src/tools/report_ablations.py --preset src/configs/ablation-smoke.toml
python src/tools/inspect_retrieval_failure.py --preset src/configs/ablation-smoke.toml --question-index 3
```

CPU execution is suitable for tests, reports, and inspection. Segmentation,
dense retrieval, answering, and evaluation are intended for a GPU allocation.

## Wormulon and Slurm flow

The normal Windows entry point is:

```powershell
.\run_on_wormulon.ps1 ablation-smoke
.\run_on_wormulon.ps1 ablation-full
```

What the PowerShell runner does:

1. Resolves `~/meeting-qa-chunking` on `koko2725@olympus.dsv.su.se`.
2. Loads the selected TOML locally and obtains its meetings and output paths.
3. Replaces the remote `src/` snapshot, preventing stale deleted files from
   surviving an upload.
4. Uploads every selected QMSum meeting, including the smoke meeting.
5. Submits generic `src/wormulon/ablation.slurm` with the preset path and
   task-specific wall time;
6. monitors the live job with `squeue` and falls back to `sacct` after it leaves
   the queue (temporary accounting failures are retried);
7. downloads the Slurm log and, on success, stages the result download before
   replacing the local result directory;
8. generates local Markdown reports; run-scoped segmentations arrive with the
   result directory.

Useful runner options:

```powershell
.\run_on_wormulon.ps1 ablation-smoke -DryRun
.\run_on_wormulon.ps1 ablation-full -NoWait
.\run_on_wormulon.ps1 ablation-full -ExistingJobId 1234
```

`-NoWait` prints the job ID and exits. `-ExistingJobId` resumes monitoring and
fetching without uploading or submitting again.

Inside Slurm, `run_preset` translates the TOML into stage commands. Each stage
is a separate `srun` process, so its GPU memory is released before the next
model is loaded. `run_python.sh`
creates/reuses `.venv-wormulon`, installs only the exact dependencies needed by
the current stage, exports `PYTHONPATH=.../src`, prints `nvidia-smi`, and fails
early if PyTorch cannot see CUDA.

The smoke job is `Bed002` with a four-hour limit. The full job uses 20 meetings
and 142 questions
from QMSum's validation/development split, excluding `Bed002` and adding the
next seed-42 candidate, `education_18`. It has a ten-hour limit. All
stages are meeting-resumable, so a second submission reuses valid saved files
and cached model calls if the first job reaches its wall-time limit.

## Caches and invalidation

Cache keys include the inputs that affect their value:

| Cache | Key includes | Safe reason to delete |
|---|---|---|
| `.cache/lumber/` | boundary model, revision, settings, prompt | rerun segmentation calls |
| `.cache/embeddings/` | dense model, revision, chunk texts | force chunk re-embedding |
| `.cache/answers/` | answer model, revision, settings, prompt | force answer generation |
| `.cache/judgments/` | judge model, revision, settings, prompt | force re-judging |

Every stage JSON contains provenance with its resolved effective configuration,
config hash, input-file hashes, input hash, and final fingerprint. A stage is
reused only when that fingerprint matches and its structure validates. The
dependency chain is:

```text
meeting -> segmentation
meeting + segmentation -> retrieval
meeting + retrieval + segmentation -> retrieved answers
meeting -> oracle answers
meeting + answer artifacts -> evaluation
```

Raw upstream files are hashed, so hand edits are detected. Prompt text, model
revision, generation settings, BM25/RRF parameters, chunk limits, and evidence
ordering all participate in the appropriate fingerprint. The complete preset
hash is stored for auditing but not used to invalidate unrelated stages.

Do not hand-edit result JSON to make a run resume. Either keep a complete valid
meeting record or remove that meeting's file and let the stage regenerate it.

## Common failures

- **`sacct` says “Resource temporarily unavailable”**: the runner retries and
  still checks `squeue`; use `-ExistingJobId` if the local connection ended.
- **SSH disconnects while the job runs**: the Slurm job survives. Reconnect with
  `-ExistingJobId <id>` to fetch it.
- **CUDA out of memory**: confirm the previous stage is a separate `srun` and
  that BERTScore is released before the judge. The 32B and 70B checkpoints are
  already 4-bit.
- **Hugging Face unauthenticated warning**: it affects download rate limits,
  not model correctness. Set `HF_TOKEN` on the cluster if downloads fail.
- **A completed result is unexpectedly reused**: compare its
  `provenance.fingerprint`; remove only that stage's meeting file if it is
  genuinely stale.
- **Local script cannot import `meeting_qa_chunking`**: set
  `$env:PYTHONPATH = "src"` before invoking a file below `src/stages` or
  `src/tools`.

## Changing the experiment safely

- New prompt: edit the relevant `.txt` file in `prompts/`; do not put prompt
  variants back into Python.
- New answer model: add a pinned `ModelSpec` in `config.py`, then select its tag
  in the TOML preset. Slurm does not change.
- New retrieval condition: change the TOML lists or extend the small validated
  registries in `config.py`. Slurm does not change.
- Token-budget ablation: add it as a new chunk/evidence accounting method. Do
  not silently reinterpret the current `w512`/`w1024` word conditions.
- New dataset: implement a loader that produces the same small `Meeting`,
  `Turn`, and `Question` structures; keep dataset-specific parsing out of the
  chunking and model modules.

Before a full run, run the unit tests, run `ablation-smoke`, inspect its Slurm
log, regenerate `report.md`/`review.md`, and manually read several examples.

## Offline package references

The curated local copies are indexed in [`vendor/README.md`](vendor/README.md).
They cover only APIs and commands used here, which keeps the bundle small enough
to browse offline.
