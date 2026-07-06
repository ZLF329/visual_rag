#!/usr/bin/env bash
set -euo pipefail

export SFT_PROJECT_DIR="${SFT_PROJECT_DIR:-/scratch/punim0614/lifuzhang/visual_rag_agent}"
cd "$SFT_PROJECT_DIR"

: "${MIMO_API_KEY:?Set MIMO_API_KEY before running}"
: "${MIMO_TRAIN_BALANCED_OUTPUT_ROOT:?Set MIMO_TRAIN_BALANCED_OUTPUT_ROOT to the original balanced run root}"

TOPUP_RUN_ID="${SFT_MULTI_TOPUP_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MULTI_OFFSET="${SFT_MULTI_OFFSET:-1000}"
MULTI_CANDIDATES="${SFT_MULTI_CANDIDATES:-800}"
MULTI_TARGET="${SFT_MULTI_TARGET:-600}"
BALANCED_LABEL="${SFT_BALANCED_LABEL:-1200}"

DATA_ROOT="${SFT_MULTI_TOPUP_DATA_ROOT:-$SFT_PROJECT_DIR/data/corpora/slidevqa_train_multi_topup_offset_${MULTI_OFFSET}_count_${MULTI_CANDIDATES}_${TOPUP_RUN_ID}}"
INDEX_DIR="${SFT_MULTI_TOPUP_INDEX:-$SFT_PROJECT_DIR/data/indexes/slidevqa_train_multi_topup_offset_${MULTI_OFFSET}_count_${MULTI_CANDIDATES}_${TOPUP_RUN_ID}}"
RUN_ROOT="$MIMO_TRAIN_BALANCED_OUTPUT_ROOT"
MULTI_OUTPUT_DIR="${SFT_MULTI_OUTPUT_DIR:-$RUN_ROOT/multi}"

python scripts/prepare_slidevqa_train_balanced.py \
  --output "$DATA_ROOT" \
  --single-target 0 \
  --multi-target "$MULTI_CANDIDATES" \
  --multi-offset "$MULTI_OFFSET" \
  --strategy "${SFT_BALANCE_STRATEGY:-packed}" \
  --clean

python scripts/build_index.py \
  --corpus "$DATA_ROOT/pages" \
  --output "$INDEX_DIR" \
  --model "${SFT_RETRIEVER_MODEL:-/scratch/punim0614/lifuzhang/models/Qwen3-VL-Embedding-8B}" \
  --attn-implementation "${SFT_RETRIEVER_ATTN:-flash_attention_2}" \
  --batch-size "${SFT_INDEX_BATCH_SIZE:-16}"

env \
  SFT_INDEX_DIR="$INDEX_DIR" \
  SFT_PAGE_ROOT="$DATA_ROOT/pages" \
  SFT_DATASET_FILE="$DATA_ROOT/train_multi.jsonl" \
  SFT_RETRIEVAL_TOP_K="${SFT_RETRIEVAL_TOP_K:-1}" \
  SFT_MAX_RETRIEVAL_STEPS="${SFT_MAX_RETRIEVAL_STEPS:-5}" \
  SFT_CONCURRENCY="${SFT_CONCURRENCY:-8}" \
  SFT_TARGET_KEPT="$MULTI_TARGET" \
  SFT_MAX_SAMPLES="$MULTI_CANDIDATES" \
  SFT_TRAJECTORY_OUTPUT_DIR="$MULTI_OUTPUT_DIR" \
  scripts/run_mimo_sft_generation.sh

export RUN_ROOT BALANCED_LABEL MULTI_TARGET
python - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
label = os.environ.get("BALANCED_LABEL", "1200")
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
    "single_target": 600,
    "multi_target": int(os.environ["MULTI_TARGET"]),
    "total_target": 600 + int(os.environ["MULTI_TARGET"]),
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
