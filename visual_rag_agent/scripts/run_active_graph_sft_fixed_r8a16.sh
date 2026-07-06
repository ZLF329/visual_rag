#!/usr/bin/env bash
set -euo pipefail

cd /scratch/punim0614/lifuzhang/visual_rag_agent

PYTHON=${PYTHON:-python3}
MODEL_PATH=${MODEL_PATH:-/scratch/punim0614/lifuzhang/models/Qwen3-VL-4B-Instruct}
DATA_DIR=${DATA_DIR:-outputs/active_graph_sft_final_1229_cropdup2_qwenzoom_881single_348multi_20260602}
RUN_ID=${RUN_ID:-active_graph_qwen3vl4b_sft_1229_qwenzoom_cropdup2_bs2ga8_r8a16_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/checkpoints/${RUN_ID}}
LOG_DIR=${LOG_DIR:-logs}
LOG_PATH=${LOG_PATH:-${LOG_DIR}/${RUN_ID}.log}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_DISABLE_PROGRESS_BARS=${HF_HUB_DISABLE_PROGRESS_BARS:-1}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}

printf 'run_id=%s
output_dir=%s
log_path=%s
data_dir=%s
lora_r=8
lora_alpha=16
per_device_train_batch_size=2
gradient_accumulation_steps=8
effective_batch_size=16
'   "${RUN_ID}" "${OUTPUT_DIR}" "${LOG_PATH}" "${DATA_DIR}"

"${PYTHON}" scripts/train_active_graph_sft_qwen_vl.py   --model-path "${MODEL_PATH}"   --train-file "${DATA_DIR}/train_sft_dynamic_chat.jsonl"   --eval-file "${DATA_DIR}/eval_sft_dynamic_chat.jsonl"   --output-dir "${OUTPUT_DIR}"   --epochs 1   --learning-rate 2e-5   --weight-decay 0.0   --warmup-ratio 0.03   --per-device-train-batch-size 2   --per-device-eval-batch-size 1   --gradient-accumulation-steps 8   --max-length 8192   --dtype bfloat16   --attn-implementation flash_attention_2   --device cuda   --logging-steps 10   --save-steps 200   --eval-steps 200   --lora-r 8   --lora-alpha 16   --lora-dropout 0.05   --lora-target-modules q_proj,k_proj,v_proj,o_proj   2>&1 | tee "${LOG_PATH}"
