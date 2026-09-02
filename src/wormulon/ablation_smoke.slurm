#!/bin/bash

# Slurm directives: one GPU allocation, four-hour smoke-test wall time.
#SBATCH --job-name=ablation-smoke
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --time=04:00:00
#SBATCH --output=slurm-ablation-smoke-%j.out

set -euo pipefail

runner="src/wormulon/run_python.sh"
root="runs/ablations/smoke"

# Separate srun processes release each stage's GPU memory before the next load.

# Stage 1: create semantic chunks once; later runs validate and reuse them.
srun bash "$runner" src/stages/ablation_segment.py \
    --meetings Bed002 \
    --output-dir runs/lumber/qmsum

# Stage 2: fixed/Lumber x dense/BM25/hybrid x two evidence budgets.
srun bash "$runner" src/stages/ablation_retrieval.py \
    --meetings Bed002 \
    --output-dir "$root/retrieval"

# Stage 3a: oracle evidence isolates answer-model capacity.
srun bash "$runner" src/stages/ablation_answer.py \
    --source oracle \
    --meetings Bed002 \
    --output-dir "$root/answers/oracle-7b" \
    --model-tag qwen2.5-7b \
    --model Qwen/Qwen2.5-7B-Instruct \
    --revision a09a35458c702b33eeacc393d103063234e8bc28

srun bash "$runner" src/stages/ablation_answer.py \
    --source oracle \
    --meetings Bed002 \
    --output-dir "$root/answers/oracle-14b" \
    --model-tag qwen2.5-14b \
    --model Qwen/Qwen2.5-14B-Instruct \
    --revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8

srun bash "$runner" src/stages/ablation_answer.py \
    --source oracle \
    --meetings Bed002 \
    --output-dir "$root/answers/oracle-32b-bnb4" \
    --model-tag qwen2.5-32b-bnb4 \
    --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit \
    --revision aa79e3472818bdec779075d80928602591d9f2a0 \
    --prequantized

# Stage 3b: the 14B model answers every retrieval condition.
srun bash "$runner" src/stages/ablation_answer.py \
    --source retrieval \
    --meetings Bed002 \
    --retrieval-dir "$root/retrieval" \
    --output-dir "$root/answers/retrieval-14b" \
    --model-tag qwen2.5-14b \
    --model Qwen/Qwen2.5-14B-Instruct \
    --revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8

# Stage 4: BERTScore first, then the 1--3 Llama judge.
srun bash "$runner" src/stages/ablation_evaluate.py \
    --answers-root "$root/answers" \
    --output-dir "$root/evaluation"

# Collect all small stage summaries into one top-level file.
srun bash "$runner" src/tools/summarize_ablations.py --root "$root"
