#!/bin/bash
# Serve Qwen3-VL-235B teacher, TP=4. gpu-mem-util 0.80 leaves ~19G/card so the
# retriever's 8B embedder can co-locate on GPU0 during generation.
# VLLM_USE_FLASHINFER_SAMPLER=0: flashinfer 0.6.12 check_cuda_arch() wrongly rejects sm_120 (Blackwell).
source /root/autodl-tmp/vllm_env/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve /root/autodl-tmp/Qwen3-VL-235B-A22B-Thinking-FP8 \
  --served-model-name qwen3vl-teacher \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --mm-encoder-tp-mode data --limit-mm-per-prompt '{"video":0}' \
  --max-model-len 32768 --gpu-memory-utilization 0.72 --max-num-seqs 32 \
  --trust-remote-code --async-scheduling \
  --host 0.0.0.0 --port 8000
