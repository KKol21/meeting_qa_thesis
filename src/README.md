# Active experiment

The active ablation stages live in `stages/`:

1. `stages/ablation_segment.py` creates missing Lumber segmentations once.
2. `stages/ablation_retrieval.py` evaluates the retrieval grid.
3. `stages/ablation_answer.py` runs an answer model over oracle or retrieved evidence.
4. `stages/ablation_evaluate.py` adds BERTScore and 1--3 LLM judgments.
5. `tools/summarize_ablations.py` collects the stage summaries.
6. `tools/report_ablations.py --preset configs/ablation-full.toml` creates `report.md` and
   the first 10 examples per condition in `review.md`.

`meeting_qa_chunking.run_preset` reads one TOML file and launches these stages
as separate processes. The preset is the experiment authority; Slurm only
provides resources. Shared implementation lives in the
`meeting_qa_chunking/` package, presets live in `configs/`, and the generic
cluster wrapper lives in `wormulon/`. The complete walkthrough is
`../docs/PIPELINE.md`.
