#!/bin/bash

#SBATCH --job-name=lumber-five
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm-lumber-five-%j.out

set -euo pipefail

srun bash steps/wormulon_lumber_smoke.sh \
    steps/11_call_lumber_boundaries.py \
    --meeting Bed002 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --revision a09a35458c702b33eeacc393d103063234e8bc28 \
    --target-tokens 550 \
    --max-new-tokens 256 \
    --max-boundaries 5
