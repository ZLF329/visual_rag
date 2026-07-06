#!/usr/bin/env bash
#SBATCH --job-name=ag25-sftvc
#SBATCH --partition=gpu-a100
#SBATCH --account=punim0614
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/punim0614/lifuzhang/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/punim0614/lifuzhang/slurm_logs/%x-%j.err
set -euo pipefail
module load Miniconda3/23.10.0-1
module load CUDA/12.4.1
export CACHE_ROOT=${CACHE_ROOT:-/scratch/punim0614/.cache}
export CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-$CACHE_ROOT/conda/envs/verl-gspo}
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TMPDIR="$CACHE_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p /scratch/punim0614/lifuzhang/slurm_logs "$TMPDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PREFIX"
export PROJECT=/scratch/punim0614/lifuzhang/visual_rag_agent
export PYTHON="$CONDA_ENV_PREFIX/bin/python"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export MODEL_PATH=/scratch/punim0614/lifuzhang/models/Qwen2.5-VL-3B-Instruct
export SFT_DATA="$PROJECT/data/sft_active_graph_qwen3vl235b_20260618/sft_pixels.jsonl"
export IMAGE_ROOT="$PROJECT/data/sft_active_graph_qwen3vl235b_20260618"
export PIPELINE_STAMP=${PIPELINE_STAMP:-$(date +%Y%m%d_%H%M%S)}
export RUN_NAME=${RUN_NAME:-qwen25vl3b_sft_NEWSM_manualcrop_VISION_spartan_${PIPELINE_STAMP}}
export OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT/outputs/checkpoints/$RUN_NAME}
export LOG_DIR="$PROJECT/logs/sft_runs"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MAX_LENGTH=${MAX_LENGTH:-16384}
export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
export NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
export MAX_STEPS=${MAX_STEPS:--1}
export LEARNING_RATE=${LEARNING_RATE:-2e-4}
export WARMUP_RATIO=${WARMUP_RATIO:-0.03}
export LOGGING_STEPS=${LOGGING_STEPS:-5}
export SAVE_STEPS=${SAVE_STEPS:-50}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}
export LORA_R=${LORA_R:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_DROPOUT=${LORA_DROPOUT:-0.05}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
export REPORT_TO=${REPORT_TO:-tensorboard}
cd "$PROJECT"
"$PYTHON" - <<'PY'
import json, re, os
from pathlib import Path
p=Path(os.environ['SFT_DATA'])
rows=bad=0
for line in p.open(encoding='utf-8'):
    row=json.loads(line); rows+=1
    msgs=row.get('messages') or []
    target=str(row.get('target') or '').strip()
    if not msgs or not target: bad+=1  # trainer appends target as assistant turn
    if 'valid_actions:' in target.lower() or '<answer>final</answer>' in target: bad+=1  # leak guard on TARGET only
print('sft_rows', rows, 'bad', bad, 'data', p)
if bad: raise SystemExit('SFT structural validation failed')
PY
REPO_ROOT="$PROJECT" MODEL_PATH="$MODEL_PATH" PYTHON="$PYTHON" bash scripts/run_active_graph_sft_qwen25vl_3b.sh
