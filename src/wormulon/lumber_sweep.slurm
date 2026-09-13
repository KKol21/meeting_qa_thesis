#!/bin/bash

# Resources such as wall time, job name, and output path are passed to sbatch.
#SBATCH --partition=gpu
#SBATCH --account=slurm-students

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch ... lumber_sweep.slurm <preset.toml>" >&2
    exit 2
fi

preset="$1"
root="runs/ablations/lumber-sweep"
targets=(500 750 1000 1250 1500)

# A fresh process per target releases model memory while cached valid meetings resume.
for target in "${targets[@]}"; do
    srun bash src/wormulon/run_python.sh src/stages/ablation_segment.py \
        --preset "$preset" \
        --target-tokens "$target" \
        --output-dir "$root/segmentation/$target"
done

srun bash src/wormulon/run_python.sh src/tools/sweep_lumber_targets.py \
    --preset "$preset" \
    --segmentation-root "$root/segmentation" \
    --targets "${targets[@]}" \
    --output "$root/sweep.json"
