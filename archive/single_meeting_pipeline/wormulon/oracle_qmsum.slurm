#!/bin/bash

#SBATCH --job-name=oracle14b-bed002
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm-oracle14b-bed002-%j.out

set -euo pipefail

srun bash steps/wormulon/run_python.sh \
    steps/oracle_qmsum.py \
    --meeting Bed002 \
    --output runs/oracle/qmsum/Bed002-14b.json \
    --model Qwen/Qwen2.5-14B-Instruct \
    --revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
    --max-new-tokens 512
