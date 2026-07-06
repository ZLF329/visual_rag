#!/usr/bin/env bash
set -euo pipefail

export SFT_PROJECT_DIR="${SFT_PROJECT_DIR:-/scratch/punim0614/lifuzhang/visual_rag_agent}"
cd "$SFT_PROJECT_DIR"

: "${MIMO_API_KEY:?Set MIMO_API_KEY before running}"

DATA_ROOT="${SLIDEVQA_TRAIN_BALANCED_ROOT:-$SFT_PROJECT_DIR/data/corpora/slidevqa_train_balanced_2000}"
INDEX_DIR="${SLIDEVQA_TRAIN_BALANCED_INDEX:-$SFT_PROJECT_DIR/data/indexes/slidevqa_train_balanced_2000}"
SFT_SINGLE_TARGET_VALUE="${SFT_SINGLE_TARGET:-1000}"
SFT_MULTI_TARGET_VALUE="${SFT_MULTI_TARGET:-1000}"
SFT_BALANCED_TARGET_TOTAL="$((SFT_SINGLE_TARGET_VALUE + SFT_MULTI_TARGET_VALUE))"
SFT_BALANCED_LABEL="${SFT_BALANCED_LABEL:-$SFT_BALANCED_TARGET_TOTAL}"
RUN_ROOT="${MIMO_TRAIN_BALANCED_OUTPUT_ROOT:-$SFT_PROJECT_DIR/outputs/sft_trajectories/mimo25_train_balanced_${SFT_BALANCED_LABEL}_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT SFT_SINGLE_TARGET_VALUE SFT_MULTI_TARGET_VALUE SFT_BALANCED_TARGET_TOTAL SFT_BALANCED_LABEL

if [[ ! -f "$DATA_ROOT/train_single.jsonl" || ! -f "$DATA_ROOT/train_multi.jsonl" ]]; then
  python scripts/prepare_slidevqa_train_balanced.py \
    --output "$DATA_ROOT" \
    --single-target "$SFT_SINGLE_TARGET_VALUE" \
    --multi-target "$SFT_MULTI_TARGET_VALUE" \
    --strategy "${SFT_BALANCE_STRATEGY:-packed}" \
    --clean
fi

if [[ ! -f "$INDEX_DIR/embeddings.npy" || ! -f "$INDEX_DIR/filenames.json" ]]; then
  python scripts/build_index.py \
    --corpus "$DATA_ROOT/pages" \
    --output "$INDEX_DIR" \
    --model "${SFT_RETRIEVER_MODEL:-/scratch/punim0614/lifuzhang/models/Qwen3-VL-Embedding-8B}" \
    --attn-implementation "${SFT_RETRIEVER_ATTN:-flash_attention_2}" \
    --batch-size "${SFT_INDEX_BATCH_SIZE:-16}"
fi

common_env=(
  SFT_INDEX_DIR="$INDEX_DIR"
  SFT_PAGE_ROOT="$DATA_ROOT/pages"
  SFT_RETRIEVAL_TOP_K="${SFT_RETRIEVAL_TOP_K:-1}"
  SFT_MAX_RETRIEVAL_STEPS="${SFT_MAX_RETRIEVAL_STEPS:-5}"
  SFT_CONCURRENCY="${SFT_CONCURRENCY:-8}"
)

env "${common_env[@]}" \
  SFT_DATASET_FILE="$DATA_ROOT/train_single.jsonl" \
  SFT_TARGET_KEPT="$SFT_SINGLE_TARGET_VALUE" \
  SFT_MAX_SAMPLES="${SFT_SINGLE_MAX_SAMPLES:-1000}" \
  SFT_TRAJECTORY_OUTPUT_DIR="$RUN_ROOT/single" \
  scripts/run_mimo_sft_generation.sh

env "${common_env[@]}" \
  SFT_DATASET_FILE="$DATA_ROOT/train_multi.jsonl" \
  SFT_TARGET_KEPT="$SFT_MULTI_TARGET_VALUE" \
  SFT_MAX_SAMPLES="${SFT_MULTI_MAX_SAMPLES:-1000}" \
  SFT_TRAJECTORY_OUTPUT_DIR="$RUN_ROOT/multi" \
  scripts/run_mimo_sft_generation.sh

python - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
label = os.environ.get("SFT_BALANCED_LABEL", "balanced")
out = run_root / f"kept_balanced_{label}.jsonl"
calls_out = run_root / f"kept_balanced_{label}_sft_calls.jsonl"
rows = []
calls = []
for name in ["single", "multi"]:
    path = run_root / name / "kept_sft_trajectories.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["source_hop_type"] = name
            rows.append(row)
    calls_path = run_root / name / "kept_sft_calls.jsonl"
    if calls_path.exists():
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            call = json.loads(line)
            call["source_hop_type"] = name
            calls.append(call)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with calls_out.open("w", encoding="utf-8") as f:
    for call in calls:
        f.write(json.dumps(call, ensure_ascii=False) + "\n")
summary = {
    "run_root": str(run_root),
    "merged_kept_file": str(out),
    "merged_kept_sft_calls_file": str(calls_out),
    "single_target": int(os.environ["SFT_SINGLE_TARGET_VALUE"]),
    "multi_target": int(os.environ["SFT_MULTI_TARGET_VALUE"]),
    "total_target": int(os.environ["SFT_BALANCED_TARGET_TOTAL"]),
    "single_kept": sum(1 for row in rows if row.get("source_hop_type") == "single"),
    "multi_kept": sum(1 for row in rows if row.get("source_hop_type") == "multi"),
    "total_kept": len(rows),
    "single_kept_sft_calls": sum(1 for call in calls if call.get("source_hop_type") == "single"),
    "multi_kept_sft_calls": sum(1 for call in calls if call.get("source_hop_type") == "multi"),
    "total_kept_sft_calls": len(calls),
}
(run_root / "balanced_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY
