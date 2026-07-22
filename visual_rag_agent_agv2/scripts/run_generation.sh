#!/bin/bash
# Usage: DATASET=sft_dataset_multi.jsonl TARGET=800 N=836 ./run_generation.sh
set -e
cd /root/autodl-tmp/visual_rag_agent
export PYTHONPATH=/root/autodl-tmp/visual_rag_agent
export TEACHER_API_KEY=dummy
export DEEPSEEK_API_KEY=YOUR_DEEPSEEK_API_KEY
export CUDA_VISIBLE_DEVICES=0          # embedder on GPU0; teacher served on all 4 via HTTP
export ACTIVE_GRAPH_RETRIEVE_K=5
DATASET="${DATASET:-sft_dataset.jsonl}"; N="${N:-2000}"; START="${START:-0}"
TARGET="${TARGET:-0}"; WORKERS="${WORKERS:-16}"
/root/autodl-tmp/vllm_env/bin/python scripts/generate_active_graph_sft_trajectories.py \
  --dataset-file /root/autodl-tmp/$DATASET \
  --config config/teacher_qwen3vl235b_sft.yaml \
  --index /root/autodl-tmp/index/sft_2000 \
  --output-dir /root/autodl-tmp/outputs/active_graph_sft_qwen3vl235b \
  --start-index $START --num-samples $N --workers $WORKERS \
  --judge deepseek --require-judge-correct --require-all-reference-pages \
  --target-kept $TARGET --stop-on-api-error
