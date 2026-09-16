#!/bin/bash

# Resources such as wall time, job name, and output path are passed to sbatch.
#SBATCH --partition=gpu
#SBATCH --account=slurm-students

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch ... baseline_sweep.slurm <preset.toml>" >&2
    exit 2
fi

srun bash src/wormulon/run_python.sh src/tools/sweep_baseline_sizes.py \
    --preset "$1" \
    --sizes 128 256 512 \
    --output runs/ablations/baseline-sweep/sweep.json
