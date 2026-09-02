#!/bin/bash

#SBATCH --job-name=lumber-smoke
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm-lumber-%j.out

set -euo pipefail

srun bash steps/wormulon_lumber_smoke.sh
