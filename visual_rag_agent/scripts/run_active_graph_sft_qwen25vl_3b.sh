#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-${PROJECT:-/scratch/punim0614/lifuzhang/visual_rag_agent}}
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -n "${CONDA_ENV_PREFIX:-}" && -x "$CONDA_ENV_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_ENV_PREFIX/bin/python"
  else
    PYTHON=python3
  fi
fi
MODEL_PATH=${MODEL_PATH:-/scratch/punim0614/lifuzhang/models/Qwen2.5-VL-3B-Instruct}
SFT_DATA=${SFT_DATA:-$REPO_ROOT/data/sft/active_graph_current/train.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-$REPO_ROOT}
RUN_NAME=${RUN_NAME:-qwen25vl3b_active_graph_sft_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/checkpoints/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs/sft_runs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$SFT_DATA" ]]; then
  cat >&2 <<EOF
ERROR: SFT_DATA does not exist: $SFT_DATA

Put the active-graph SFT JSONL at this path or launch with:
  SFT_DATA=/path/to/train.jsonl bash scripts/run_active_graph_sft_qwen25vl_3b.sh

Expected rows contain either:
  - messages: [{role, content}, ..., {role: assistant, content: ...}]
  - conversations: ShareGPT-style rows
  - prompt + response/target/completion
EOF
  exit 3
fi

cd "$REPO_ROOT"
echo "run_name=$RUN_NAME"
echo "model=$MODEL_PATH"
echo "sft_data=$SFT_DATA"
echo "output=$OUTPUT_DIR"
echo "log=$LOG_FILE"

"$PYTHON" scripts/train_active_graph_sft_qwen25vl.py \
  --model-path "$MODEL_PATH" \
  --train-data "$SFT_DATA" \
  --image-root "$IMAGE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --max-length "${MAX_LENGTH:-16384}" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --max-steps "${MAX_STEPS:--1}" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --warmup-ratio "${WARMUP_RATIO:-0.03}" \
  --logging-steps "${LOGGING_STEPS:-5}" \
  --save-steps "${SAVE_STEPS:-50}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-3}" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-32}" \
  --lora-dropout "${LORA_DROPOUT:-0.05}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --report-to "${REPORT_TO:-tensorboard}" \
  ${EXTRA_SFT_ARGS:-} 2>&1 | tee "$LOG_FILE"
