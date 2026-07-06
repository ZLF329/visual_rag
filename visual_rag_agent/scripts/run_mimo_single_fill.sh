#!/usr/bin/env bash
set -euo pipefail

export SFT_PROJECT_DIR="${SFT_PROJECT_DIR:-/scratch/punim0614/lifuzhang/visual_rag_agent}"
cd "$SFT_PROJECT_DIR"

: "${MIMO_API_KEY:?Set MIMO_API_KEY before running}"
: "${MIMO_TRAIN_BALANCED_OUTPUT_ROOT:?Set MIMO_TRAIN_BALANCED_OUTPUT_ROOT to the original balanced run root}"
: "${SFT_SINGLE_FILL_TARGET:?Set SFT_SINGLE_FILL_TARGET to the single-hop fill count}"

FILL_RUN_ID="${SFT_SINGLE_FILL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SINGLE_OFFSET="${SFT_SINGLE_OFFSET:-1000}"
SINGLE_CANDIDATES="${SFT_SINGLE_CANDIDATES:-300}"
BALANCED_LABEL="${SFT_BALANCED_LABEL:-1200}"

DATA_ROOT="${SFT_SINGLE_FILL_DATA_ROOT:-$SFT_PROJECT_DIR/data/corpora/slidevqa_train_single_fill_offset_${SINGLE_OFFSET}_count_${SINGLE_CANDIDATES}_${FILL_RUN_ID}}"
INDEX_DIR="${SFT_SINGLE_FILL_INDEX:-$SFT_PROJECT_DIR/data/indexes/slidevqa_train_single_fill_offset_${SINGLE_OFFSET}_count_${SINGLE_CANDIDATES}_${FILL_RUN_ID}}"
RUN_ROOT="$MIMO_TRAIN_BALANCED_OUTPUT_ROOT"
FILL_OUTPUT_DIR="${SFT_SINGLE_FILL_OUTPUT_DIR:-$RUN_ROOT/single_fill}"

python scripts/prepare_slidevqa_train_balanced.py \
  --output "$DATA_ROOT" \
  --single-target "$SINGLE_CANDIDATES" \
  --single-offset "$SINGLE_OFFSET" \
  --multi-target 0 \
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
  SFT_DATASET_FILE="$DATA_ROOT/train_single.jsonl" \
  SFT_RETRIEVAL_TOP_K="${SFT_RETRIEVAL_TOP_K:-1}" \
  SFT_MAX_RETRIEVAL_STEPS="${SFT_MAX_RETRIEVAL_STEPS:-5}" \
  SFT_CONCURRENCY="${SFT_CONCURRENCY:-8}" \
  SFT_TARGET_KEPT="$SFT_SINGLE_FILL_TARGET" \
  SFT_MAX_SAMPLES="$SINGLE_CANDIDATES" \
  SFT_TRAJECTORY_OUTPUT_DIR="$FILL_OUTPUT_DIR" \
  scripts/run_mimo_sft_generation.sh

export RUN_ROOT BALANCED_LABEL SFT_SINGLE_FILL_TARGET
python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
label = os.environ.get("BALANCED_LABEL", "1200")
out = run_root / f"kept_balanced_{label}.jsonl"
calls_out = run_root / f"kept_balanced_{label}_sft_calls.jsonl"

sources = [
    ("single", "single"),
    ("multi", "multi"),
    ("single_fill", "single"),
]
rows = []
calls = []
source_counts = Counter()
call_source_counts = Counter()
for dirname, hop_type in sources:
    path = run_root / dirname / "kept_sft_trajectories.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["source_hop_type"] = hop_type
            row["source_bucket"] = dirname
            rows.append(row)
            source_counts[dirname] += 1
    calls_path = run_root / dirname / "kept_sft_calls.jsonl"
    if calls_path.exists():
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            call = json.loads(line)
            call["source_hop_type"] = hop_type
            call["source_bucket"] = dirname
            calls.append(call)
            call_source_counts[dirname] += 1

with out.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with calls_out.open("w", encoding="utf-8") as f:
    for call in calls:
        f.write(json.dumps(call, ensure_ascii=False) + "\n")

keys = [(str(row.get("deck_name") or ""), str(row.get("question") or "").strip()) for row in rows]
duplicate_keys = sum(count - 1 for count in Counter(keys).values() if count > 1)
sample_ids = [str(row.get("sample_id")) for row in rows if row.get("sample_id") is not None]
duplicate_sample_ids = sum(count - 1 for count in Counter(sample_ids).values() if count > 1)

summary = {
    "run_root": str(run_root),
    "merged_kept_file": str(out),
    "merged_kept_sft_calls_file": str(calls_out),
    "total_target": int(label) if str(label).isdigit() else 1200,
    "single_original_kept": source_counts["single"],
    "single_fill_kept": source_counts["single_fill"],
    "single_kept": source_counts["single"] + source_counts["single_fill"],
    "multi_kept": source_counts["multi"],
    "total_kept": len(rows),
    "single_original_kept_sft_calls": call_source_counts["single"],
    "single_fill_kept_sft_calls": call_source_counts["single_fill"],
    "single_kept_sft_calls": call_source_counts["single"] + call_source_counts["single_fill"],
    "multi_kept_sft_calls": call_source_counts["multi"],
    "total_kept_sft_calls": len(calls),
    "duplicate_deck_question_pairs": duplicate_keys,
    "duplicate_sample_ids": duplicate_sample_ids,
    "fill_hop_type": "single",
    "fill_target": int(os.environ["SFT_SINGLE_FILL_TARGET"]),
}
(run_root / "balanced_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
PY
