#!/bin/bash
cd /root/autodl-tmp
export HF_HOME=/root/autodl-tmp/hf
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES=${VLLM_DEVICES:-0}
MODEL=${MODEL_PATH:-/root/autodl-tmp/models/qwen25vl3b_ag_sft_verify_noimg-merged}
SERVED=${SERVED_NAME:-qwen25vl3b-verify}
exec /root/autodl-tmp/envs/verl/bin/vllm serve \
  "$MODEL" \
  --port ${PORT:-8000} \
  --served-model-name "$SERVED" \
  --gpu-memory-utilization ${GMEM:-0.62} \
  --tensor-parallel-size ${TP:-1} \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image":8}' \
  --trust-remote-code
