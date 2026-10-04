# Meeting QA chunking

This repository contains a QMSum experiment comparing complete-turn packing,
strict word packing, and a LumberChunker adaptation. Validation artifacts also
retain the exploratory single-turn baseline. ELITR-Bench is not implemented;
whether it remains in thesis scope is an open decision.

## Layout

- `src/`: the active experiment, package, presets, and Wormulon jobs
- `src/stages/`: the four experiment stages, in execution order
- `src/tools/`: reporting and manual-inspection commands
- `src/meeting_qa_chunking/`: reusable experiment implementation
- `src/configs/`: smoke, validation, and held-out test presets and manifests
- `docs/PIPELINE.md`: offline code and Slurm walkthrough
- `docs/vendor/`: curated offline dependency documentation
- `data/`: local source data (ignored by Git)
- `runs/`: fetched experiment results (new files are ignored; selected ablation
  results are committed)

## Ablation workflow

The final retrieval grid contains 27 conditions: turn-packed/word-packed/Lumber
chunks, dense/BM25/hybrid retrieval, and 512/1024/2048-word evidence budgets. Oracle answers compare
Qwen2.5 7B, 14B, and a 32B bitsandbytes 4-bit checkpoint. End-to-end answers
use 14B across all 27 retrieval conditions. Every saved answer is evaluated
with BERTScore and a 4-bit Llama 3.3 70B judge on a 1--3 scale: invalid/incorrect,
partially correct, or correct.

First run the one-meeting smoke test. It exercises all retrieval methods and
all three answer models, including the quantized 32B backend:

```powershell
.\run_on_wormulon.ps1 ablation-smoke
```

The completed 20-meeting validation experiment and its selected artifacts live
under `runs/ablations/validation-full`. It can be resumed with:

```powershell
.\run_on_wormulon.ps1 ablation-validation-full
```

After freezing all choices, run the resumable 35-meeting, 244-question held-out
test experiment. It excludes the exploratory single-turn baseline:

```powershell
.\run_on_wormulon.ps1 ablation-full
```

To select Lumber's target window on the development meetings, run the
500/750/1000/1250/1500 sweep. It evaluates dense, BM25, and hybrid retrieval
at 512, 1024, and 2048 evidence words, then downloads `sweep.json` and
`sweep.md`:

```powershell
.\run_on_wormulon.ps1 lumber-sweep
```

Run a five-meeting retrieval-only diagnostic comparing 7B, 14B, and
quantized 32B Lumber boundary models at the selected target:

```powershell
.\run_on_wormulon.ps1 lumber-model-check
```

Run the retrieval-only 128/256/512-word sensitivity sweep for both packed
baselines:

```powershell
.\run_on_wormulon.ps1 baseline-sweep
```

Compare the selected 1,000-target Lumber segmentation with ten
geometry-matched random turn-boundary partitions, using retrieval metrics only:

```powershell
.\run_on_wormulon.ps1 boundary-control
```

Compare exact evidence clipping with dropping or fully including the final
chunk using an existing retrieval run (no models required):

```powershell
$env:PYTHONPATH = "src"
python src/tools/analyze_clipping_sensitivity.py `
    --preset src/configs/ablation-validation-full.toml `
    --output runs/ablations/validation-full/clipping-sensitivity.json
```

Add meeting-level percentile bootstrap intervals and paired contrasts to a
completed run (no models required):

```powershell
$env:PYTHONPATH = "src"
python src/tools/analyze_uncertainty.py `
    --preset src/configs/ablation-validation-full.toml `
    --output runs/ablations/validation-full/uncertainty.json
```

Relate added retrieval coverage at 2,048 rather than 1,024 evidence words to
paired answer-score changes. This writes meeting-cluster bootstrap intervals
to Markdown and every question-level delta to the sibling JSON:

```powershell
$env:PYTHONPATH = "src"
python src/tools/analyze_budget_deltas.py `
    --preset src/configs/ablation-full.toml `
    --output runs/ablations/full/budget-deltas.json
```

The TOML preset controls meetings, models, parameters, and output paths. These
commands derive their selected QMSum files from that preset, upload them with
`src/`, wait for Slurm, and download the complete result directory.

Use `-NoWait` to submit without waiting, or check paths without connecting:

```powershell
.\run_on_wormulon.ps1 ablation-smoke -DryRun
```

Inspect a saved retrieval failure locally without loading a model:

```powershell
$env:PYTHONPATH = "src"
python src/tools/inspect_retrieval_failure.py `
    --preset src/configs/ablation-smoke.toml `
    --question-index 3
```

Export one retrieval condition against the 14B oracle for manual review:

```powershell
python src/tools/export_review.py `
    --run validation-full `
    --condition lumber__dense__w512
```

Create the qualitative-review workbook. It samples ten questions from each of
three query types and compares turn-packed with Lumber at 2,048 words under
dense retrieval (60 annotation rows):

```powershell
$env:PYTHONPATH = "src"
python src/tools/export_qualitative_workbook.py `
    --preset src/configs/ablation-full.toml `
    --output runs/ablations/full/qualitative-review.xlsx `
    --retriever dense --questions-per-type 10 --seed 42
```

Generate the thesis-ready result figures from the saved test and validation
artifacts (PNG for review, PDF for typesetting):

```powershell
pip install -e ".[plots]"
python src/tools/plot_results.py
```

The figure index and suggested placement are documented in
[`docs/RESULT_VISUALIZATIONS.md`](docs/RESULT_VISUALIZATIONS.md); the underlying
calculations and artifact provenance are explained in
[`docs/PLOT_GENERATION.md`](docs/PLOT_GENERATION.md).

For the complete data flow, caches, commands, failure recovery, and Slurm
explanation, read [`docs/PIPELINE.md`](docs/PIPELINE.md).
