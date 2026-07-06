#!/usr/bin/env bash
set -euo pipefail

export SFT_PROJECT_DIR="${SFT_PROJECT_DIR:-/scratch/punim0614/lifuzhang/visual_rag_agent}"
cd "$SFT_PROJECT_DIR"
export PYTHONPATH="$SFT_PROJECT_DIR:${PYTHONPATH:-}"

set -a
[ -f .secrets/qwen36plus.env ] && source .secrets/qwen36plus.env
# Prefer the user-supplied DashScope key when present.
[ -f .secrets/qwen36plus_user.env ] && source .secrets/qwen36plus_user.env
[ -n "${SFT_CROP_TOPUP_EXTRA_ENV:-}" ] && [ -f "$SFT_CROP_TOPUP_EXTRA_ENV" ] && source "$SFT_CROP_TOPUP_EXTRA_ENV"
[ -f .secrets/deepseek.env ] && source .secrets/deepseek.env
set +a

: "${MIMO_API_KEY:?Set MIMO_API_KEY or .secrets/qwen36plus.env before running}"
: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY or .secrets/deepseek.env before running}"

RUN_ID="${SFT_CROP_TOPUP_RUN_ID:-crop_multi_$(date +%Y%m%d_%H%M%S)}"
RUN_BASE="${SFT_CROP_TOPUP_RUN_BASE:-$SFT_PROJECT_DIR/outputs/active_graph_crop_topup_multi_$RUN_ID}"
DATA_ROOT="${SFT_CROP_TOPUP_DATA_ROOT:-$SFT_PROJECT_DIR/data/corpora/slidevqa_train_active_graph_topup_balanced2000_remaining}"
DATASET_FILE="${SFT_CROP_TOPUP_DATASET_FILE:-$DATA_ROOT/train_multi.jsonl}"
INDEX_DIR="${SFT_CROP_TOPUP_INDEX_DIR:-$SFT_PROJECT_DIR/data/indexes/slidevqa_train_active_graph_topup_balanced2000_remaining}"
SHARDS="${SFT_CROP_TOPUP_SHARDS:-3}"
WORKERS="${SFT_CROP_TOPUP_WORKERS:-8}"
TARGET_SCORE100="${SFT_CROP_TOPUP_TARGET_SCORE100:-500}"
SEED_RUN_BASES="${SFT_CROP_TOPUP_SEED_RUN_BASES:-$SFT_PROJECT_DIR/outputs/active_graph_multi_score100_seed_with5641clean_20260602}"
MONITOR_INTERVAL="${SFT_CROP_TOPUP_MONITOR_INTERVAL:-10}"
TOTAL_SAMPLES="${SFT_CROP_TOPUP_TOTAL_SAMPLES:-$(wc -l < "$DATASET_FILE")}"
START_INDEX="${SFT_CROP_TOPUP_START_INDEX:-0}"
RETRIEVAL_TOP_K="${SFT_RETRIEVAL_TOP_K:-1}"
MAX_ITERS="${SFT_CROP_TOPUP_MAX_ITERS:-15}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
API_BASE_URL="${MIMO_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

MODELS=(
  "qwen3.5-397b-a17b"
  "qwen3.5-122b-a10b"
  "qwen3.6-27b"
  "qwen3-VL-235B-A22B-instruct"
  "qwen3.5-flash"
  "qwen3.5-flash-2026-02-23"
  "qwen3.6-flash"
  "qwen3.6-flash-2026-04-16"
  "qwen3.7-plus-2026-05-26"
  "qwen3.7-plus"
)

if [ -n "${SFT_CROP_TOPUP_MODELS:-}" ]; then
  read -r -a MODELS <<< "$SFT_CROP_TOPUP_MODELS"
fi

mkdir -p "$RUN_BASE/configs" "$RUN_BASE/logs"

if [ ! -f "$INDEX_DIR/embeddings.npy" ] || [ ! -f "$INDEX_DIR/filenames.json" ]; then
  echo "[index] building $INDEX_DIR from $DATA_ROOT/pages"
  python scripts/build_index.py \
    --corpus "$DATA_ROOT/pages" \
    --output "$INDEX_DIR" \
    --model "${SFT_RETRIEVER_MODEL:-/scratch/punim0614/lifuzhang/models/Qwen3-VL-Embedding-8B}" \
    --attn-implementation "${SFT_RETRIEVER_ATTN:-flash_attention_2}" \
    --batch-size "${SFT_INDEX_BATCH_SIZE:-16}"
else
  echo "[index] reusing $INDEX_DIR"
fi

printf '%s\n' "${MODELS[@]}" > "$RUN_BASE/model_order.txt"
{
  echo "run_id=$RUN_ID"
  echo "run_base=$RUN_BASE"
  echo "dataset_file=$DATASET_FILE"
  echo "index_dir=$INDEX_DIR"
  echo "start_index=$START_INDEX"
  echo "total_samples=$TOTAL_SAMPLES"
  echo "shards=$SHARDS"
  echo "workers_per_shard=$WORKERS"
  echo "target_score100=$TARGET_SCORE100"
  echo "seed_run_bases=$SEED_RUN_BASES"
} > "$RUN_BASE/launch_plan.env"

quota_pattern="stopped_on_api_error.*true\|quota\|balance\|free quota\|免费额度\|rate limit\|rate_limit\|too many requests\|insufficient\|model not found\|no permission\|forbidden\|access denied\|unauthorized\|billing"

audit_all() {
  local out_dir="$RUN_BASE/score100_selected_all"
  local args=(--run-base "$RUN_BASE" --output-dir "$out_dir")
  local seed
  for seed in $SEED_RUN_BASES; do
    if [ -d "$seed" ]; then
      args=(--run-base "$seed" "${args[@]}")
    fi
  done
  python scripts/audit_active_graph_multi_score100.py "${args[@]}" >/tmp/active_graph_crop_topup_audit_${RUN_ID}.json
  python - "$out_dir/summary.json" <<'PYCOUNT'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
print(summary.get("score100_selected", 0))
PYCOUNT
}

kill_shards() {
  local pid
  for pid in "$@"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  sleep 2
  for pid in "$@"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  done
}

current_score100="$(audit_all)"
echo "[audit] initial_score100=$current_score100 target=$TARGET_SCORE100"
if [ "$current_score100" -ge "$TARGET_SCORE100" ]; then
  echo "[done] target already reached before generation"
  echo "[done] run_base=$RUN_BASE"
  echo "[done] selected=$RUN_BASE/score100_selected_all/multi_score100_crop_allowed.jsonl"
  exit 0
fi

make_config() {
  local model="$1"
  local safe="$2"
  local cfg="$RUN_BASE/configs/${safe}.yaml"
  python - "$cfg" "$model" "$INDEX_DIR" "$API_BASE_URL" "$MAX_ITERS" "$RETRIEVAL_TOP_K" "$RUN_BASE" "$safe" <<'PYCONFIG'
import sys
from pathlib import Path

cfg, model, index_dir, api_base_url, max_iters, top_k, run_base, safe = sys.argv[1:]
lines = [
    "models:",
    "  vlm:",
    "    provider: openai_compatible",
    f"    name: {model}",
    "    env_files:",
    "      - .secrets/qwen36plus.env",
    f"    api_base_url: {api_base_url}",
    "    api_key_env: MIMO_API_KEY",
    "    api_timeout: 180",
    "    api_max_retries: 2",
    "    api_image_quality: 90",
    "    max_tokens: 1024",
    "    temperature: 0.0",
    "    prompt_mode: chat",
    "    policy_output_format: json",
    "    api_policy_response_format: json_object",
    "  retriever:",
    "    name: /scratch/punim0614/lifuzhang/models/Qwen3-VL-Embedding-8B",
    f"    index_path: {index_dir}",
    "",
    "agent:",
    f"  top_k: {top_k}",
    f"  max_iters: {max_iters}",
    "  partial_memory_capacity: 2",
    "",
    "image_budget:",
    "  yes_pixels: 400000",
    "  partial_pixels: 400000",
    "",
    "dataset:",
    "  name: slidevqa",
    "  split: train",
    "  num_samples: 0",
    "",
    "runtime:",
    f"  output_dir: {Path(run_base) / safe}",
    "  device: cuda",
    "  dtype: bfloat16",
    "  attn_implementation: flash_attention_2",
    "",
]
Path(cfg).write_text("\n".join(lines), encoding="utf-8")
PYCONFIG
  echo "$cfg"
}

shard_size=$(( (TOTAL_SAMPLES + SHARDS - 1) / SHARDS ))
for model in "${MODELS[@]}"; do
  safe="$(echo "$model" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/_/g')"
  cfg="$(make_config "$model" "$safe")"
  echo "[model] $model cfg=$cfg"
  pids=()
  for shard in $(seq 0 $((SHARDS - 1))); do
    start=$(( START_INDEX + shard * shard_size ))
    end=$(( START_INDEX + TOTAL_SAMPLES ))
    if [ "$start" -ge "$end" ]; then
      continue
    fi
    num="$shard_size"
    if [ $((start + num)) -gt "$end" ]; then
      num=$(( end - start ))
    fi
    out_dir="$RUN_BASE/$safe/shard${shard}_${start}_$((start + num - 1))"
    log="$RUN_BASE/logs/${safe}_shard${shard}.log"
    echo "[launch] model=$model shard=$shard start=$start num=$num workers=$WORKERS out=$out_dir"
    (
      PYTHONPATH=. python scripts/generate_active_graph_sft_trajectories.py \
        --dataset-file "$DATASET_FILE" \
        --config "$cfg" \
        --output-dir "$out_dir" \
        --start-index "$start" \
        --num-samples "$num" \
        --window-size "${SFT_CROP_TOPUP_WINDOW_SIZE:-2}" \
        --index "$INDEX_DIR" \
        --judge deepseek \
        --judge-model "$DEEPSEEK_MODEL" \
        --judge-timeout "${SFT_JUDGE_TIMEOUT:-60}" \
        --judge-max-retries "${SFT_JUDGE_MAX_RETRIES:-2}" \
        --require-judge-correct \
        --require-all-reference-pages \
        --stop-on-api-error \
        --workers "$WORKERS"
    ) > "$log" 2>&1 &
    pids+=("$!")
  done

  status=0
  switched=0
  while true; do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" >/dev/null 2>&1; then
        alive=1
      fi
    done
    if grep -R "$quota_pattern" "$RUN_BASE/logs/${safe}_shard"*.log >/dev/null 2>&1; then
      echo "[switch] detected quota/api/model error for $model; killing current shards and moving to next model"
      kill_shards "${pids[@]}"
      switched=1
      status=1
      break
    fi
    if [ "$alive" -eq 0 ]; then
      break
    fi
    sleep "$MONITOR_INTERVAL"
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid" 2>/dev/null; then
      status=1
    fi
  done
  echo "[model_done] $model status=$status switched=$switched"

  python scripts/audit_active_graph_multi_score100.py \
    --run-base "$RUN_BASE/$safe" \
    --output-dir "$RUN_BASE/$safe/score100_selected" || true

  current_score100="$(audit_all)"
  echo "[audit] after_model=$model score100=$current_score100 target=$TARGET_SCORE100"
  if [ "$current_score100" -ge "$TARGET_SCORE100" ]; then
    echo "[done] target reached"
    break
  fi
done

current_score100="$(audit_all)"

echo "[done] run_base=$RUN_BASE"
echo "[done] selected=$RUN_BASE/score100_selected_all/multi_score100_crop_allowed.jsonl"
echo "[done] score100=$current_score100 target=$TARGET_SCORE100"
