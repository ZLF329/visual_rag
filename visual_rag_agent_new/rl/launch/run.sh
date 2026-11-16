#!/bin/bash
set -euo pipefail
prep=$(cd "$(dirname "$0")/.." && pwd)
export WORK_ROOT=/opt/visual-rag-local
code=/opt/rl1200_prefill_forced_20261111/verl-agent
export VISUAL_RAG_AGENT_PATH="$prep/runtime/visual_rag_agent_new"
export PYTHONPATH="$code:$VISUAL_RAG_AGENT_PATH"
export PATH="$WORK_ROOT/envs/rl/bin:$PATH"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export ROLLOUT_ACTIVE_SUBSET=1 VLLM_FLASH_ATTN_VERSION=2 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_V1=0
export ACTIVE_GRAPH_RETRIEVE_K=5 SLIDEVQA_STEP_WORKERS=32
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 RAY_agent_register_timeout_ms=180000 RAY_worker_register_timeout_seconds=180
export RAY_prestart_worker_first_driver=0 RAY_enable_worker_prestart=0 RAY_worker_maximum_startup_concurrency=8
export TRITON_CACHE_DIR=/tmp/prefill_g/triton TORCHINDUCTOR_CACHE_DIR=/tmp/prefill_g/inductor VLLM_CACHE_ROOT=/tmp/prefill_g/vllm
export RAY_TMPDIR=/tmp/prefill_g_ray
source /root/.config/visual-rag/credentials.env
: "${DEEPSEEK_API_KEY:?DeepSeek credential missing}"
export DEEPSEEK_MODEL=deepseek-v4-flash DEEPSEEK_BASE_URL=https://api.deepseek.com
cd "$code"
exec "$WORK_ROOT/envs/rl/bin/python" "$prep/deployment/train.py" "$1"
