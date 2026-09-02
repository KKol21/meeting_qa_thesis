#!/bin/bash

#SBATCH --job-name=retrieval-bed002
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm-retrieval-bed002-%j.out

set -euo pipefail

srun bash steps/wormulon/run_python.sh \
    steps/compare_retrieval.py \
    --meeting Bed002 \
    --lumber-result runs/lumber/qmsum/Bed002.json \
    --output runs/retrieval/qmsum/Bed002.json \
    --fixed-chunk-words 256 \
    --evidence-words 512
