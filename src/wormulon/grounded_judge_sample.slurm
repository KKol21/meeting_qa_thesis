#!/bin/bash

# Resources such as wall time, job name, and output path are passed to sbatch.
#SBATCH --partition=gpu
#SBATCH --account=slurm-students

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: grounded_judge_sample.slurm <preset.toml>" >&2
    exit 2
fi

srun bash src/wormulon/run_python.sh src/tools/judge_qualitative_sample.py \
    --preset "$1"
