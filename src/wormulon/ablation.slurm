#!/bin/bash

# Resources such as wall time, job name, and output path are passed to sbatch.
#SBATCH --partition=gpu
#SBATCH --account=slurm-students

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch ... ablation.slurm <preset.toml>" >&2
    exit 2
fi

# The preset runner starts every GPU-heavy stage in its own srun process.
PYTHONPATH=src python3 -m meeting_qa_chunking.run_preset \
    --preset "$1" \
    --use-srun
