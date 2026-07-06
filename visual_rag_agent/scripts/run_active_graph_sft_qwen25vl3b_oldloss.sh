#!/usr/bin/env bash
set -euo pipefail

cd /scratch/punim0614/lifuzhang/visual_rag_agent

export MODEL_PATH=${MODEL_PATH:-/scratch/punim0614/lifuzhang/models/Qwen2.5-VL-3B-Instruct}
export DATA_DIR=${DATA_DIR:-outputs/active_graph_sft_final_1229_cropdup2_qwenzoom_881single_348multi_20260602_portable}
export RUN_ID=${RUN_ID:-active_graph_qwen25vl3b_sft_oldloss_1229_qwenzoom_$(date +%Y%m%d_%H%M%S)}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/checkpoints/${RUN_ID}}
export LOG_PATH=${LOG_PATH:-logs/${RUN_ID}.log}

# Important: this wrapper intentionally calls the original SFT trainer/script.
# Loss contract stays unchanged: sliding-window messages are prompt-only;
# labels are masked before prompt_len, so loss is only on the current target.
exec bash scripts/run_active_graph_sft_fixed_r8a16.sh
