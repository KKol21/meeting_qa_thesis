# Offline dependency references

These are local snapshots of official documentation selected for this
repository, downloaded on 2026-08-29. Open the files directly in a browser or
Markdown preview. Some HTML styling and navigation requires the internet, but
the article text is embedded in each file.

| Used for | Repository version | Local copy | Official online source |
|---|---:|---|---|
| Transformers generation and `generate` | 5.15.1 | [Generation](transformers-generation.html) | [Hugging Face](https://huggingface.co/docs/transformers/v5.15.1/en/main_classes/text_generation) |
| Transformers/bitsandbytes integration | Transformers 5.15.1, bitsandbytes 0.50.2 | [Bitsandbytes integration](transformers-bitsandbytes.html) | [Hugging Face](https://huggingface.co/docs/transformers/v5.15.1/en/quantization/bitsandbytes) |
| `SentenceTransformer` and `encode` | 5.7.0 | [SentenceTransformer API](sentence-transformers.html) | [Sentence Transformers](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html) |
| Accelerate large-model placement | 1.14.0 | [Working with large models](accelerate-big-modeling.html) | [Hugging Face](https://huggingface.co/docs/accelerate/v1.14.0/en/package_reference/big_modeling) |
| BERTScore usage and interpretation | 0.3.13 | [Project README](bert-score-readme.md) | [Tiiiger/bert_score](https://github.com/Tiiiger/bert_score) |
| ROUGE scorer package | 0.1.2 | [Project README](rouge-score-readme.md) | [google-research/rouge](https://github.com/google-research/google-research/tree/master/rouge) |
| Submit a batch job | cluster-installed Slurm | [`sbatch`](slurm-sbatch.html) | [SchedMD](https://slurm.schedmd.com/sbatch.html) |
| Start a job step | cluster-installed Slurm | [`srun`](slurm-srun.html) | [SchedMD](https://slurm.schedmd.com/srun.html) |
| Inspect live jobs | cluster-installed Slurm | [`squeue`](slurm-squeue.html) | [SchedMD](https://slurm.schedmd.com/squeue.html) |
| Inspect accounting/completed jobs | cluster-installed Slurm | [`sacct`](slurm-sacct.html) | [SchedMD](https://slurm.schedmd.com/sacct.html) |

PyTorch is supplied by Wormulon's CUDA environment rather than pinned in
`pyproject.toml`, so its exact build is cluster-dependent. `run_python.sh`
checks `torch.cuda.is_available()` and prints the assigned GPU before a stage
starts.

These snapshots are reference material only; they are not imported or used at
runtime.
