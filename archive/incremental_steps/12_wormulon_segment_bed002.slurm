#!/bin/bash

#SBATCH --job-name=lumber-bed002
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm-lumber-bed002-%j.out

set -euo pipefail

srun bash steps/wormulon_lumber_smoke.sh \
    steps/12_segment_qmsum_meeting.py \
    --meeting Bed002 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --revision a09a35458c702b33eeacc393d103063234e8bc28 \
    --target-tokens 550 \
    --max-new-tokens 256
