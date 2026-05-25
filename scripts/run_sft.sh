#!/usr/bin/env bash
set -xeuo pipefail

cd "$(dirname "$0")/.."

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-VL-4B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-outputs/sft_trajectories/gpt5mini_20260525_004442/kept_sft_trajectories.jsonl}
PARQUET_ROOT=${PARQUET_ROOT:-/root/autodl-tmp/hf_data/NTT-hil-insight-SlideVQA/data}
IMAGE_ROOT=${IMAGE_ROOT:-data/corpora/slidevqa_sft/pages}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/checkpoints/sft_qwen3vl4b_visor}

python3 scripts/materialize_sft_images.py \
  --train-file "$TRAIN_FILE" \
  --parquet-root "$PARQUET_ROOT" \
  --split train \
  --output-root "$IMAGE_ROOT"

PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
python3 scripts/train_sft_qwen_vl.py \
  --model-path "$MODEL_PATH" \
  --train-file "$TRAIN_FILE" \
  --image-root "$IMAGE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 2 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.1 \
  --weight-decay 0.01 \
  --max-length 16384 \
  --dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --logging-steps 5 \
  --save-steps 200 \
  --save-total-limit 3
