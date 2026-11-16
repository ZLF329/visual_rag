#!/usr/bin/env bash
# Install onto a regional network volume. The supplied source bundle contains no credentials.
# This prepares dependencies and validates this single GPU; it does not launch training.
set -euo pipefail

WORK_ROOT=${WORK_ROOT:-/workspace}
export WORK_ROOT
VRA_ROOT=${VRA_ROOT:-$WORK_ROOT/code/visual_rag_agent_new}
ENV_ROOT=$WORK_ROOT/envs/rl
PREP_ROOT=$WORK_ROOT/preparation
export UV_CACHE_DIR=$WORK_ROOT/cache/uv
export UV_PYTHON_INSTALL_DIR=$WORK_ROOT/python
export UV_DEFAULT_INDEX=https://pypi.org/simple
export UV_LINK_MODE=copy
export PIP_CACHE_DIR=$WORK_ROOT/cache/pip
export HF_HOME=$WORK_ROOT/cache/huggingface
export XDG_CACHE_HOME=$WORK_ROOT/cache
export TMPDIR=$WORK_ROOT/cache/tmp
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$PREP_ROOT" "$TMPDIR" "$WORK_ROOT/envs"
rm -f "$PREP_ROOT/environment_setup.exit"
trap 'rc=$?; printf "%s\n" "$rc" > "$PREP_ROOT/environment_setup.exit"' EXIT
UV_BIN=${UV_BIN:-/usr/bin/uv}

if [ ! -x "$ENV_ROOT/bin/python" ]; then
  "$UV_BIN" python install 3.12
  "$UV_BIN" venv --managed-python --python 3.12 --relocatable "$ENV_ROOT"
fi
"$UV_BIN" pip install --python "$ENV_ROOT/bin/python" -r "$VRA_ROOT/rl/experiments/runpod-requirements.txt"
# SlideVQA uses image inputs; PyAV is available for video decoding. Remove the
# optional Decord wheel, whose platform metadata fails uv's compatibility check.
"$UV_BIN" pip uninstall --python "$ENV_ROOT/bin/python" decord

# Match the actual torch C++ ABI, rather than assuming the wheel's ABI variant.
FLASH_WHEEL_URL=$(
  "$ENV_ROOT/bin/python" - <<'PY'
import sys, torch
abi = 'TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE'
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
name = f'flash_attn-2.8.3+cu12torch2.8cxx11abi{abi}-{tag}-{tag}-linux_x86_64.whl'
print('https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/' + name)
PY
)
"$UV_BIN" pip install --python "$ENV_ROOT/bin/python" --no-deps "$FLASH_WHEEL_URL"
"$UV_BIN" pip check --python "$ENV_ROOT/bin/python"
"$UV_BIN" pip freeze --python "$ENV_ROOT/bin/python" > "$PREP_ROOT/requirements-resolved.txt"

"$ENV_ROOT/bin/python" - <<'PY'
import json
from pathlib import Path
import os
import torch, transformers, vllm, ray, flash_attn, tensordict
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from flash_attn import flash_attn_func
torch.manual_seed(0)
q, k, v = [torch.randn(1, 32, 2, 64, device='cuda', dtype=torch.float16) for _ in range(3)]
actual = flash_attn_func(q, k, v, causal=True)
expected = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
report = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__,
          'vllm': vllm.__version__, 'ray': ray.__version__, 'flash_attn': flash_attn.__version__,
          'tensordict': tensordict.__version__, 'visible_gpus': torch.cuda.device_count(),
          'gpu': torch.cuda.get_device_name(0), 'flash_attention_matches_pytorch_sdpa': True,
          'multi_gpu_nccl_validated': False, 'training_validated': False}
Path(os.environ['WORK_ROOT'] + '/preparation/environment_validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
PY
cd "$VRA_ROOT"
"$ENV_ROOT/bin/python" tests/run_tests.py > "$PREP_ROOT/agent_tests.log" 2>&1
tail -3 "$PREP_ROOT/agent_tests.log"
"$ENV_ROOT/bin/python" scripts/runpod_preflight.py > "$PREP_ROOT/prepared_inventory.json"
printf 'DEPENDENCY_ENVIRONMENT_READY\n'
