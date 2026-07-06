#!/usr/bin/env bash
set -euo pipefail

export SFT_PROJECT_DIR="${SFT_PROJECT_DIR:-/scratch/punim0614/lifuzhang/visual_rag_agent}"
cd "$SFT_PROJECT_DIR"

: "${MIMO_API_KEY:?Set MIMO_API_KEY before running}"

export PYTHONPATH="$SFT_PROJECT_DIR:${PYTHONPATH:-}"
export MIMO_BASE_URL="${MIMO_BASE_URL:-https://api.xiaomimimo.com/v1}"
export MIMO_RESPONSE_FORMAT="${MIMO_RESPONSE_FORMAT:-json_object}"
export MIMO_THINKING="${MIMO_THINKING:-disabled}"

export MAN_SFT_GENERATOR_MODEL="${MAN_SFT_GENERATOR_MODEL:-mimo-v2.5}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
export SFT_DATASET_FILE="${SFT_DATASET_FILE:-$SFT_PROJECT_DIR/data/corpora/slidevqa/test.jsonl}"
export SFT_PAGE_ROOT="${SFT_PAGE_ROOT:-$SFT_PROJECT_DIR/data/corpora/slidevqa/pages}"
export SFT_RETRIEVER_MODEL="${SFT_RETRIEVER_MODEL:-/scratch/punim0614/lifuzhang/models/Qwen3-VL-Embedding-8B}"
export SFT_RETRIEVER_ATTN="${SFT_RETRIEVER_ATTN:-flash_attention_2}"
export SFT_CONCURRENCY="${SFT_CONCURRENCY:-1}"
export SFT_MAX_SAMPLES="${SFT_MAX_SAMPLES:-1}"
export SFT_TARGET_KEPT="${SFT_TARGET_KEPT:-1}"
export SFT_RUN_ID="${SFT_RUN_ID:-mimo25_$(date +%Y%m%d_%H%M%S)}"
export SFT_TRAJECTORY_OUTPUT_DIR="${SFT_TRAJECTORY_OUTPUT_DIR:-$SFT_PROJECT_DIR/outputs/sft_trajectories/$SFT_RUN_ID}"

python scripts/run_sft_generation.py
