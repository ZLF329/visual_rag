#!/usr/bin/env bash
set -euo pipefail

cd /scratch/punim0614/lifuzhang/visual_rag_agent

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
export MODEL_PATH=${MODEL_PATH:-/scratch/punim0614/lifuzhang/models/Qwen2.5-VL-3B-Instruct}
export DATA_DIR=${DATA_DIR:-outputs/active_graph_sft_final_1229_cropdup2_qwenzoom_881single_348multi_20260602_portable}
export DATASET_FILE=${DATASET_FILE:-data/corpora/slidevqa/test.jsonl}
export INDEX_DIR=${INDEX_DIR:-data/indexes/slidevqa_test_main}
export SFT_RUN_ID=${SFT_RUN_ID:-active_graph_qwen25vl3b_sft_oldloss_1229_qwenzoom_r16a32_${STAMP}}
export EVAL_RUN_ID=${EVAL_RUN_ID:-active_graph_qwen25vl3b_sft_eval500_oldloss_${STAMP}}
export CONFIG_PATH=${CONFIG_PATH:-config/${EVAL_RUN_ID}.yaml}

# This intentionally reuses the old Qwen-VL SFT trainer and old train+eval flow.
# Loss contract: sliding-window messages are prompt-only; labels before prompt_len
# are -100, so loss is only on the current policy target.
exec bash scripts/run_active_graph_r16a32_train_then_eval.sh
