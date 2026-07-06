#!/usr/bin/env bash
# Build a parallel conda env mirroring autodl's working stack:
# python3.12 + torch2.8/cu128 + vllm0.11.0 + flash_attn2.8.3 + transformers4.57.3
# Leaves verl-gspo untouched. verl-agent installed editable (--no-deps).
set -xeo pipefail

module load Miniconda3/23.10.0-1
module load CUDA/12.8.0

# compute nodes set a bare LANG=UTF-8 which python3.12 rejects (LookupError: unknown encoding)
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

export CACHE_ROOT=/scratch/punim0614/.cache
export NEW_ENV=$CACHE_ROOT/conda/envs/verl-vllm011
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TMPDIR="$CACHE_ROOT/tmp"
export TEMP="$TMPDIR"; export TMP="$TMPDIR"
mkdir -p "$CACHE_ROOT"/{conda/envs,pip,huggingface,torch,triton,tmp}

source "$(conda info --base)/etc/profile.d/conda.sh"
[ -x "$NEW_ENV/bin/python" ] || conda create -y -p "$NEW_ENV" python=3.12
conda activate "$NEW_ENV"

python -m pip install --upgrade pip setuptools wheel packaging ninja

# 1) torch stack (cu128 wheels)
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# 2) vllm 0.11.0 (torch already satisfied; pulls its own compatible deps)
python -m pip install vllm==0.11.0

# 3) pin verl/ML deps to autodl-proven versions (run AFTER vllm so these win)
python -m pip install \
  transformers==4.57.3 tokenizers==0.22.2 safetensors==0.7.0 \
  ray==2.50.0 tensordict==0.10.0 accelerate==1.13.0 datasets==4.8.5 peft==0.19.1 \
  numpy==2.2.6 pyarrow==24.0.0 hydra-core==1.3.2 codetiming==1.4.0 dill==0.4.1 \
  qwen-vl-utils==0.0.14 torchdata==0.11.0 xformers==0.0.32.post1 \
  liger-kernel pylatexenc wandb uvicorn fastapi pybind11 pandas pylatexenc

# 4) register verl (editable, no deps so it can't disturb the pinned stack)
python -m pip install -e /scratch/punim0614/lifuzhang/verl-agent --no-deps

# 5) flash-attn 2.8.3 prebuilt wheel (cu12 torch2.8 cp312, abiFALSE) - no build
python -m pip install --force-reinstall --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

# 6) verify
python - <<'PYCHK'
import numpy, torch, transformers, vllm, ray, flash_attn
import verl
print('OK torch', torch.__version__, 'cuda', torch.version.cuda)
print('OK vllm', vllm.__version__)
print('OK transformers', transformers.__version__)
print('OK flash_attn', flash_attn.__version__)
print('OK ray', ray.__version__)
print('OK verl import')
PYCHK
echo "BUILD_DONE_OK"
