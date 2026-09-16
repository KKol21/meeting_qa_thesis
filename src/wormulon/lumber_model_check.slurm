#!/bin/bash

# Resources such as wall time, job name, and output path are passed to sbatch.
#SBATCH --partition=gpu
#SBATCH --account=slurm-students

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch ... lumber_model_check.slurm <preset.toml>" >&2
    exit 2
fi

preset="$1"
root="runs/ablations/lumber-model-check"
models=(qwen2.5-7b qwen2.5-14b qwen2.5-32b-bnb4)
meeting_count=5
target=1000

# Separate processes release each boundary model before loading the next one.
for model in "${models[@]}"; do
    srun bash src/wormulon/run_python.sh src/stages/ablation_segment.py \
        --preset "$preset" \
        --model "$model" \
        --meeting-count "$meeting_count" \
        --target-tokens "$target" \
        --output-dir "$root/segmentation/$model"
done

srun bash src/wormulon/run_python.sh src/tools/check_lumber_models.py \
    --preset "$preset" \
    --segmentation-root "$root/segmentation" \
    --models "${models[@]}" \
    --reference-model qwen2.5-14b \
    --meeting-count "$meeting_count" \
    --target-tokens "$target" \
    --output "$root/check.json"
