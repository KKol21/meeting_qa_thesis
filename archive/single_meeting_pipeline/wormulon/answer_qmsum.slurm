#!/bin/bash

#SBATCH --job-name=answer-bed002
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm-answer-bed002-%j.out

set -euo pipefail

srun bash steps/wormulon/run_python.sh \
    steps/answer_qmsum.py \
    --meeting Bed002 \
    --lumber-result runs/lumber/qmsum/Bed002.json \
    --retrieval-result runs/retrieval/qmsum/Bed002.json \
    --output runs/answers/qmsum/Bed002.json \
    --model Qwen/Qwen2.5-7B-Instruct \
    --revision a09a35458c702b33eeacc393d103063234e8bc28 \
    --max-new-tokens 512
