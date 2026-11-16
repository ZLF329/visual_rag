#!/bin/bash
set -euo pipefail
prep=$(cd "$(dirname "$0")/.." && pwd)
cd "$prep"
exec 9>deployment.lock
flock -n 9
test ! -e gpu_smoke/verified.json
if ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1024 {bad=1} END {exit bad}'; then
  printf 'GPUs are occupied\n' >&2
  exit 1
fi
mkdir -p gpu_smoke
if PREFILL_GPU_SMOKE=1 bash deployment/run.sh smoke > gpu_smoke/train.log 2>&1; then
  printf '0\n' > gpu_smoke/exit
else
  code=$?
  printf '%s\n' "$code" > gpu_smoke/exit
  exit "$code"
fi
/opt/visual-rag-local/envs/rl/bin/python deployment/verify_smoke.py
run=/workspace/outputs/rl/qwen25vl7b_scoree1_rl1200_m400_prefill_g_decay05_bs16_s75_traj16_20261111
mkdir -p "$run"
test ! -e "$run/effective_config.yaml"
printf '%s\n' "$$" > "$run/supervisor.pid"
if bash deployment/run.sh formal > "$run/train.log" 2>&1; then
  printf '0\n' > "$run/training.exit"
else
  code=$?
  printf '%s\n' "$code" > "$run/training.exit"
  exit "$code"
fi
