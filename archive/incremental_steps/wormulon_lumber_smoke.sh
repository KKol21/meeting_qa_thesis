#!/bin/bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

env_dir=".venv-wormulon"
if [[ ! -x "$env_dir/bin/python" ]]; then
    python3 -m venv --system-site-packages "$env_dir"
fi

if ! "$env_dir/bin/python" -c \
    "import transformers; assert transformers.__version__ == '5.15.1'" \
    2>/dev/null; then
    "$env_dir/bin/python" -m pip install "transformers==5.15.1"
fi

if ! "$env_dir/bin/python" -c \
    "import sentence_transformers; assert sentence_transformers.__version__ == '5.7.0'" \
    2>/dev/null; then
    "$env_dir/bin/python" -m pip install "sentence-transformers==5.7.0"
fi

if [[ ! -f "data/raw/qmsum/data/ALL/val/Bed002.json" ]]; then
    echo "Missing QMSum data/raw/qmsum/data/ALL/val/Bed002.json" >&2
    exit 1
fi

nvidia-smi
"$env_dir/bin/python" -c \
    "import torch; assert torch.cuda.is_available(); print('CUDA:', torch.cuda.get_device_name())"

if [[ $# -gt 0 ]]; then
    "$env_dir/bin/python" "$@"
else
    "$env_dir/bin/python" steps/09_call_lumber_model.py \
        --meeting Bed002 \
        --model Qwen/Qwen2.5-3B-Instruct \
        --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
        --target-tokens 550 \
        --max-new-tokens 32
fi
