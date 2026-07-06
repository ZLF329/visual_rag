#!/usr/bin/env bash
#SBATCH --job-name=qwen25-sft-eval
#SBATCH --partition=gpu-a100-short
#SBATCH --account=punim0614
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:2
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

cd /scratch/punim0614/lifuzhang/visual_rag_agent

python - <<'PYCHECK'
import torch, transformers, peft
import qwen_vl_utils
print('env ok torch', torch.__version__, 'cuda', torch.version.cuda)
print('transformers', transformers.__version__)
print('cuda visible', torch.cuda.device_count())
PYCHECK

export SFT_RUN_ID=${SFT_RUN_ID:-active_graph_qwen25vl3b_sft_oldloss_1229_qwenzoom_r16a32_spartan}
export EVAL_RUN_ID=${EVAL_RUN_ID:-active_graph_qwen25vl3b_sft_eval500_oldloss_spartan}

MODEL_PATH=/scratch/punim0614/lifuzhang/models/Qwen2.5-VL-3B-Instruct PYTHON="$CONDA_ENV_PREFIX/bin/python" bash scripts/run_active_graph_qwen25vl3b_oldloss_train_then_eval.sh
