#!/bin/bash

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: run_python.sh <script> [arguments...]" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

# Reuse one small environment across stages; install only each stage's extras.
env_dir=".venv-wormulon"
if [[ ! -x "$env_dir/bin/python" ]]; then
    python3 -m venv --system-site-packages "$env_dir"
fi

if ! "$env_dir/bin/python" -c \
    "import transformers; assert transformers.__version__ == '5.15.1'" \
    2>/dev/null; then
    "$env_dir/bin/python" -m pip install "transformers==5.15.1"
fi

if [[ "$1" == "src/stages/ablation_retrieval.py" ]] && \
    ! "$env_dir/bin/python" -c \
        "import sentence_transformers; assert sentence_transformers.__version__ == '5.7.0'" \
        2>/dev/null; then
    "$env_dir/bin/python" -m pip install "sentence-transformers==5.7.0"
fi

if [[ "$1" == "src/stages/ablation_answer.py" ]] && \
    ! "$env_dir/bin/python" -c \
        "from importlib.metadata import version; assert version('rouge-score') == '0.1.2'" \
        2>/dev/null; then
    "$env_dir/bin/python" -m pip install "rouge-score==0.1.2"
fi

if [[ "$1" == "src/stages/ablation_evaluate.py" ]] && \
    ! "$env_dir/bin/python" -c \
        "from importlib.metadata import version; assert version('bert-score') == '0.3.13'" \
        2>/dev/null; then
    "$env_dir/bin/python" -m pip install "bert-score==0.3.13"
fi

if [[ "$1" == "src/stages/ablation_evaluate.py" || \
      "$1" == "src/stages/ablation_answer.py" ]] && \
    ! "$env_dir/bin/python" -c \
        "from importlib.metadata import version; assert version('accelerate') == '1.14.0'; assert version('bitsandbytes') == '0.50.2'" \
        2>/dev/null; then
    "$env_dir/bin/python" -m pip install \
        "accelerate==1.14.0" \
        "bitsandbytes==0.50.2"
fi

# Fail here, before model loading, if Slurm did not assign a usable GPU.
nvidia-smi
"$env_dir/bin/python" -c \
    "import torch; assert torch.cuda.is_available(); print('CUDA:', torch.cuda.get_device_name())"

exec "$env_dir/bin/python" "$@"
