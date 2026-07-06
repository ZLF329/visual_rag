#!/usr/bin/env bash
#SBATCH --job-name=ag-idx-800
#SBATCH --partition=gpu-a100-short
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
mkdir -p "$TMPDIR" /scratch/punim0614/lifuzhang/slurm_logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PREFIX" || true
export PATH="$CONDA_ENV_PREFIX/bin:$PATH"
export PY="$CONDA_ENV_PREFIX/bin/python"

cd /scratch/punim0614/lifuzhang/visual_rag_agent
export PYTHONPATH="/scratch/punim0614/lifuzhang/visual_rag_agent:${PYTHONPATH:-}"
"$PY" -m pip install -q qwen-vl-utils
"$PY" - <<'PYCHK'
import sys, site
print({'python': sys.executable, 'site': site.getsitepackages()})
from qwen_vl_utils.vision_process import process_vision_info
print({'qwen_vl_utils': 'ok'})
PYCHK

CORPUS=/scratch/punim0614/lifuzhang/visual_rag_agent/data/corpora/active_graph_rl_800single_400multi_20260614/pages
OUT=/scratch/punim0614/lifuzhang/visual_rag_agent/data/indexes/active_graph_rl_800single_400multi_20260614
MODEL=/scratch/punim0614/lifuzhang/models/Qwen3-VL-Embedding-8B
export TMP_OUT=${OUT}.tmp_${SLURM_JOB_ID}
rm -rf "$TMP_OUT"
mkdir -p "$TMP_OUT"

"$PY" scripts/build_index.py \
  --corpus "$CORPUS" \
  --output "$TMP_OUT" \
  --model "$MODEL" \
  --device cuda \
  --dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --batch-size 8

"$PY" - <<'PY'
import json, numpy as np
from pathlib import Path
out=Path(__import__('os').environ['TMP_OUT'])
entries=json.loads((out/'filenames.json').read_text())
emb=np.load(out/'embeddings.npy', mmap_mode='r')
print({'entries': len(entries), 'embedding_shape': emb.shape, 'first': entries[0] if entries else None})
if len(entries) < 12000:
    raise SystemExit(f'index entry count too low: {len(entries)}')
PY

rm -rf "$OUT"
mv "$TMP_OUT" "$OUT"
ln -sfn "$OUT" /scratch/punim0614/lifuzhang/active_graph_rl_workspace/data/index
