#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/root/autodl-tmp/visual_rag_agent}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-VL-4B-Instruct}
SFT_DATA=${SFT_DATA:-$REPO_ROOT/outputs/active_graph_sft_final_1229_cropdup2_qwenzoom_881single_348multi_20260602/train_sft_dynamic_chat.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-$REPO_ROOT}
RUN_NAME=${RUN_NAME:-qwen3vl4b_active_graph_sft_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/checkpoints/$RUN_NAME}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs/sft_runs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$SFT_DATA" ]]; then
  echo "ERROR: SFT_DATA does not exist: $SFT_DATA" >&2
  exit 3
fi

cd "$REPO_ROOT"
echo "run_name=$RUN_NAME"
echo "model=$MODEL_PATH"
echo "sft_data=$SFT_DATA"
echo "output=$OUTPUT_DIR"
echo "log=$LOG_FILE"

"$PYTHON" scripts/train_active_graph_sft_qwen3vl.py \
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
