#!/usr/bin/env bash
set -euo pipefail

cd /scratch/punim0614/lifuzhang/visual_rag_agent

PYTHON=${PYTHON:-python3}
MODEL_PATH=${MODEL_PATH:-/scratch/punim0614/lifuzhang/models/Qwen3-VL-4B-Instruct}
DATA_DIR=${DATA_DIR:-outputs/active_graph_sft_final_1229_cropdup2_qwenzoom_881single_348multi_20260602}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SFT_RUN_ID=${SFT_RUN_ID:-active_graph_qwen3vl4b_sft_1229_qwenzoom_cropdup2_bs2ga8_r16a32_${STAMP}}
SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-outputs/checkpoints/${SFT_RUN_ID}}
SFT_LOG_PATH=${SFT_LOG_PATH:-logs/${SFT_RUN_ID}.log}
EVAL_RUN_ID=${EVAL_RUN_ID:-active_graph_sft_eval3_1229_qwenzoom_bs2ga8_r16a32_${STAMP}}
EVAL_ROOT=${EVAL_ROOT:-outputs/${EVAL_RUN_ID}}
CONFIG_PATH=${CONFIG_PATH:-config/${EVAL_RUN_ID}.yaml}
DATASET_FILE=${DATASET_FILE:-data/corpora/slidevqa/test.jsonl}
INDEX_DIR=${INDEX_DIR:-data/indexes/slidevqa_test_main}

mkdir -p logs outputs/checkpoints "${EVAL_ROOT}" "$(dirname "${CONFIG_PATH}")"

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HUB_DISABLE_PROGRESS_BARS=${HF_HUB_DISABLE_PROGRESS_BARS:-1}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}

printf 'phase=train\nsft_run_id=%s\nsft_output_dir=%s\nsft_log_path=%s\ndata_dir=%s\nlora_r=16\nlora_alpha=32\nper_device_train_batch_size=2\ngradient_accumulation_steps=8\neffective_batch_size=16\n' \
  "${SFT_RUN_ID}" "${SFT_OUTPUT_DIR}" "${SFT_LOG_PATH}" "${DATA_DIR}"

"${PYTHON}" scripts/train_active_graph_sft_qwen_vl.py \
  --model-path "${MODEL_PATH}" \
  --train-file "${DATA_DIR}/train_sft_dynamic_chat.jsonl" \
  --eval-file "${DATA_DIR}/eval_sft_dynamic_chat.jsonl" \
  --output-dir "${SFT_OUTPUT_DIR}" \
  --epochs 1 \
  --learning-rate 2e-5 \
  --weight-decay 0.0 \
  --warmup-ratio 0.03 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-length 8192 \
  --dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --device cuda \
  --logging-steps 10 \
  --save-steps 200 \
  --eval-steps 200 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj \
  2>&1 | tee "${SFT_LOG_PATH}"

printf 'phase=eval\neval_run_id=%s\neval_root=%s\nadapter=%s\nconfig=%s\n' \
  "${EVAL_RUN_ID}" "${EVAL_ROOT}" "${SFT_OUTPUT_DIR}" "${CONFIG_PATH}"

"${PYTHON}" - <<PYCONF
from pathlib import Path
try:
    import yaml
except Exception:
    yaml = None
config = {
    'models': {
        'vlm': {
            'provider': 'qwen',
            'name': '${MODEL_PATH}',
            'adapter_path': '${SFT_OUTPUT_DIR}',
            'max_tokens': 1024,
            'temperature': 0.0,
            'prompt_mode': 'system_in_user',
        },
        'retriever': {
            'name': '/scratch/punim0614/lifuzhang/models/Qwen3-VL-Embedding-8B',
            'index_path': '${INDEX_DIR}',
        },
    },
    'agent': {'top_k': 1, 'max_iters': 12, 'partial_memory_capacity': 2},
    'image_budget': {'yes_pixels': 400000, 'partial_pixels': 400000},
    'dataset': {'name': 'slidevqa', 'split': 'test', 'num_samples': 500},
    'runtime': {
        'output_dir': '${EVAL_ROOT}',
        'device': 'cuda',
        'dtype': 'bfloat16',
        'attn_implementation': 'flash_attention_2',
    },
}
path = Path('${CONFIG_PATH}')
path.parent.mkdir(parents=True, exist_ok=True)
if yaml is not None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
else:
    def dump(obj, indent=0):
        lines=[]; sp='  '*indent
        for k,v in obj.items():
            if isinstance(v, dict):
                lines.append(f'{sp}{k}:')
                lines.extend(dump(v, indent+1))
            else:
                lines.append(f'{sp}{k}: {v}')
        return lines
    path.write_text('\n'.join(dump(config))+'\n', encoding='utf-8')
PYCONF

shards=("shard0 0 167" "shard1 167 167" "shard2 334 166")
IFS=',' read -r -a CUDA_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
pids=()
shard_idx=0
for shard in "${shards[@]}"; do
  set -- ${shard}
  name=$1; start=$2; count=$3
  out="${EVAL_ROOT}/${name}"
  mkdir -p "${out}"
  log="logs/${EVAL_RUN_ID}_${name}.log"
  gpu_id="${CUDA_IDS[$((shard_idx % ${#CUDA_IDS[@]}))]}"
  echo "launch_eval_shard name=${name} start=${start} count=${count} gpu=${gpu_id} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" scripts/evaluate.py \
    --dataset-file "${DATASET_FILE}" \
    --config "${CONFIG_PATH}" \
    --output "${out}" \
    --start-index "${start}" \
    --num-samples "${count}" \
    --baseline agent \
    --judge none \
    > "${log}" 2>&1 &
  pids+=("$!")
  shard_idx=$((shard_idx + 1))
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

"${PYTHON}" scripts/merge_eval_shards.py \
  --roots "${EVAL_ROOT}/shard0" "${EVAL_ROOT}/shard1" "${EVAL_ROOT}/shard2" \
  --output "${EVAL_ROOT}/merged" \
  --baseline agent

printf 'phase=done\nsft_output_dir=%s\neval_root=%s\n' "${SFT_OUTPUT_DIR}" "${EVAL_ROOT}"
