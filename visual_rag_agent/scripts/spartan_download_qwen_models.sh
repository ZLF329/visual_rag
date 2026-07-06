#!/usr/bin/env bash
set -euo pipefail

module load Miniconda3/23.10.0-1

export CACHE_ROOT=${CACHE_ROOT:-/scratch/punim0614/.cache}
export MODEL_ROOT=${MODEL_ROOT:-/scratch/punim0614/lifuzhang/models}
mkdir -p "$CACHE_ROOT"/{huggingface,torch,triton,tmp,python-userbase} "$MODEL_ROOT"

export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TMPDIR="$CACHE_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export PYTHONUSERBASE="$CACHE_ROOT/python-userbase"
export PATH="$PYTHONUSERBASE/bin:$PATH"
PY_VER=$(python - <<'PYVER'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PYVER
)
export PYTHONPATH="$PYTHONUSERBASE/lib/python${PY_VER}/site-packages:${PYTHONPATH:-}"

python - <<'PYCHECK' || python -m pip install --user --upgrade 'huggingface_hub[hf_transfer]'
import huggingface_hub
print('huggingface_hub', huggingface_hub.__version__)
PYCHECK

export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

python - <<'PYDL'
from pathlib import Path
from huggingface_hub import snapshot_download

model_root = Path(__import__('os').environ['MODEL_ROOT'])
models = {
    'Qwen/Qwen2.5-VL-3B-Instruct': model_root / 'Qwen2.5-VL-3B-Instruct',
    'Qwen/Qwen3-VL-4B-Instruct': model_root / 'Qwen3-VL-4B-Instruct',
    'Qwen/Qwen3-VL-Embedding-8B': model_root / 'Qwen3-VL-Embedding-8B',
}
for repo_id, local_dir in models.items():
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f'[download] {repo_id} -> {local_dir}', flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        resume_download=True,
        local_dir_use_symlinks=False,
    )
    print(f'[done] {repo_id}', flush=True)
PYDL

printf '\nModel directories:\n'
du -sh "$MODEL_ROOT"/Qwen2.5-VL-3B-Instruct "$MODEL_ROOT"/Qwen3-VL-4B-Instruct "$MODEL_ROOT"/Qwen3-VL-Embedding-8B
