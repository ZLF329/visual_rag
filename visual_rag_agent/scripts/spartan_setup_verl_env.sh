#!/usr/bin/env bash
set -euo pipefail

module load Miniconda3/23.10.0-1
module load CUDA/12.4.1

export CACHE_ROOT=${CACHE_ROOT:-/scratch/punim0614/.cache}
export CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-$CACHE_ROOT/conda/envs/verl-gspo}
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TMPDIR="$CACHE_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$CACHE_ROOT"/{conda/envs,pip,huggingface,torch,triton,tmp}

if [ ! -x "$CONDA_ENV_PREFIX/bin/python" ]; then
  conda create -y -p "$CONDA_ENV_PREFIX" python=3.10
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PREFIX"
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
python -m pip install vllm==0.8.4
REQ_NO_FLASH="$TMPDIR/verl_requirements_no_flash.txt"
awk 'tolower($1) !~ /^flash-attn/' /scratch/punim0614/lifuzhang/verl-agent/requirements.txt > "$REQ_NO_FLASH"
python -m pip install -r "$REQ_NO_FLASH" --no-deps
python -m pip install accelerate codetiming datasets dill hydra-core liger-kernel numpy pandas peft 'pyarrow>=19.0.0' pybind11 pylatexenc 'ray[default]' 'tensordict<=0.6.2' torchdata wandb 'packaging>=20.0' uvicorn fastapi 'qwen-vl-utils[decord]' transformers==4.51.1
python -m pip install -e /scratch/punim0614/lifuzhang/verl-agent
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
python -m pip install --force-reinstall --no-deps "$FLASH_ATTN_WHEEL"
python - <<'PYCHK'
import numpy, torch, transformers, vllm, ray
import verl
print('numpy', numpy.__version__)
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('transformers', transformers.__version__)
print('vllm', vllm.__version__)
print('verl import ok')
PYCHK
