#!/bin/bash

# Resources such as wall time, job name, and output path are passed to sbatch.
#SBATCH --partition=gpu
#SBATCH --account=slurm-students

set -euo pipefail

if [[ $# -lt 7 ]]; then
    echo "Usage: grounded_judge.slurm <preset> --meeting ID --question-index N --condition NAME" >&2
    exit 2
fi

preset="$1"
shift
srun bash src/wormulon/run_python.sh src/tools/judge_grounded_answer.py \
    --preset "$preset" "$@"
