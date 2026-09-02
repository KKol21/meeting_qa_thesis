#!/bin/bash

#SBATCH --job-name=segment-bed002
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm-segment-bed002-%j.out

set -euo pipefail

srun bash steps/wormulon/run_python.sh \
    steps/segment_qmsum.py \
    --meeting Bed002 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --revision a09a35458c702b33eeacc393d103063234e8bc28 \
    --target-tokens 550 \
    --max-new-tokens 256
