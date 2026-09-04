# Meeting QA chunking

This repository contains a QMSum experiment comparing complete-turn packing,
strict word packing, and Lumber-style semantic chunking. ELITR-Bench will be
added after this path is stable.

## Layout

- `src/`: the active experiment, package, presets, and Wormulon jobs
- `src/stages/`: the four experiment stages, in execution order
- `src/tools/`: reporting and manual-inspection commands
- `src/meeting_qa_chunking/`: reusable experiment implementation
- `src/configs/`: smoke/full presets and the full-run meeting list
- `docs/PIPELINE.md`: offline code and Slurm walkthrough
- `docs/vendor/`: curated offline dependency documentation
- `data/`: local source data (ignored by Git)
- `runs/`: fetched experiment results (ignored by Git)

## Ablation workflow

The retrieval grid contains 18 conditions: turn-packed/word-packed/Lumber
chunks, dense/BM25/hybrid retrieval, and 512/1024-word evidence budgets. Oracle answers compare
Qwen2.5 7B, 14B, and a 32B bitsandbytes 4-bit checkpoint. End-to-end answers
use 14B across all 18 retrieval conditions. Every saved answer is evaluated
with BERTScore and a 4-bit Llama 3.3 70B judge on a 1--3 scale: invalid/incorrect,
partially correct, or correct.

First run the one-meeting smoke test. It exercises all retrieval methods and
all three answer models, including the quantized 32B backend:

```powershell
.\run_on_wormulon.ps1 ablation-smoke
```

Only after that succeeds, run the resumable 20-meeting, 142-question
development experiment:

```powershell
.\run_on_wormulon.ps1 ablation-full
```

The TOML preset controls meetings, models, parameters, and output paths. Both
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

For the complete data flow, caches, commands, failure recovery, and Slurm
explanation, read [`docs/PIPELINE.md`](docs/PIPELINE.md).
