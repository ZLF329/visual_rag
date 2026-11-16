#!/bin/bash
# Serve the teacher with vLLM (default: Qwen3.8-27B bf16 on one H800), or the student (MODEL_PATH=... SERVED_NAME=student
# VLLM_EXTRA_ARGS="--generation-config vllm": plain greedy, without the model's repetition_penalty 1.05 default).
# gpu-memory-utilization leaves ~12 GB for the ColQwen query encoder that shares the GPU
# (its page bank lives in RAM: config/teacher.yaml bank_device: cpu).
ROOT=${ROOT:-/root/rivermind-data}
source "$ROOT/envs/vllm/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# FlashInfer's top-k/top-p sampler JIT-compiles with nvcc, which the H800 box lacks; vLLM's native
# sampler gives the same result (and the teacher decodes greedily).
export VLLM_USE_FLASHINFER_SAMPLER=0
MODEL_PATH=${MODEL_PATH:-$ROOT/models/Qwen3.8-27B}
SERVED_NAME=${SERVED_NAME:-teacher}
TP=${TP:-1}
GPU_UTIL=${GPU_UTIL:-0.83}
PORT=${PORT:-8000}                      # one replica per GPU (data parallel): GPU=0 PORT=8000, GPU=1 PORT=8001
export CUDA_VISIBLE_DEVICES=${GPU:-0}
exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --limit-mm-per-prompt '{"video":0}' \
  --max-model-len 32768 --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs 32 \
  --trust-remote-code \
  --host 127.0.0.1 --port "$PORT" ${VLLM_EXTRA_ARGS:-}
