#!/bin/bash

# Resources such as wall time, job name, and output path are passed to sbatch.
#SBATCH --partition=gpu
#SBATCH --account=slurm-students

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch ... boundary_control.slurm <preset.toml>" >&2
    exit 2
fi

srun bash src/wormulon/run_python.sh src/tools/boundary_shuffled_control.py \
    --preset "$1" \
    --segmentation-dir runs/ablations/lumber-sweep/segmentation/1000 \
    --target 1000 \
    --randomizations 10 \
    --candidates 1000 \
    --seed 42 \
    --output runs/ablations/boundary-control/control.json
