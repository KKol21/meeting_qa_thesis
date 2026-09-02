# Active experiment

The active ablation stages live in `stages/`:

1. `stages/ablation_segment.py` creates missing Lumber segmentations once.
2. `stages/ablation_retrieval.py` evaluates the retrieval grid.
3. `stages/ablation_answer.py` runs an answer model over oracle or retrieved evidence.
4. `stages/ablation_evaluate.py` adds BERTScore and 1--3 LLM judgments.
5. `tools/summarize_ablations.py` collects the stage summaries.
6. `tools/report_ablations.py --root runs/ablations/full` creates `report.md` and
   the first 10 examples per condition in `review.md`.

Shared implementation lives in the `meeting_qa_chunking/` package. Run presets
and the meeting manifest live in `configs/`; Wormulon launch files remain in
`wormulon/`. The complete walkthrough is `../docs/PIPELINE.md`.
Completed exploratory scripts are preserved under `../archive/incremental_steps/`.
The superseded single-meeting pipeline is preserved under
`../archive/single_meeting_pipeline/`.
