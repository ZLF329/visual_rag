#!/usr/bin/env bash
#SBATCH --job-name=ag4b-sft
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
module load CUDA/12.8.0
export CACHE_ROOT=${CACHE_ROOT:-/scratch/punim0614/.cache}
export CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-$CACHE_ROOT/conda/envs/verl-vllm011}
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TMPDIR="$CACHE_ROOT/tmp"
mkdir -p /scratch/punim0614/lifuzhang/slurm_logs "$TMPDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PREFIX"
export REPO_ROOT=/scratch/punim0614/lifuzhang/visual_rag_agent
export PYTHON="$CONDA_ENV_PREFIX/bin/python"
export PYTHONPATH="$REPO_ROOT"
export MODEL_PATH=/scratch/punim0614/lifuzhang/models/Qwen3-VL-4B-Instruct
export SFT_DATA="$REPO_ROOT/outputs/active_graph_sft_final_1229_cropdup2_qwenzoom_881single_348multi_20260602/train_sft_dynamic_chat_NEWSM_noevref_portable.jsonl"
export IMAGE_ROOT="$REPO_ROOT"
export RUN_NAME=qwen3vl4b_active_graph_sft_NEWSM_20260613
cd "$REPO_ROOT"
bash scripts/run_active_graph_sft_qwen3vl4b.sh
