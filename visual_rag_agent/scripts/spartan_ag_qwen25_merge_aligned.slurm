#!/usr/bin/env bash
#SBATCH --job-name=ag25-merge
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
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TMPDIR="$CACHE_ROOT/tmp"
mkdir -p /scratch/punim0614/lifuzhang/slurm_logs "$TMPDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PREFIX"
export PROJECT=/scratch/punim0614/lifuzhang/visual_rag_agent
export PYTHON="$CONDA_ENV_PREFIX/bin/python"
export BASE_MODEL=/scratch/punim0614/lifuzhang/models/Qwen2.5-VL-3B-Instruct
export PIPELINE_STAMP=${PIPELINE_STAMP:-$(date +%Y%m%d_%H%M%S)}
export RUN_NAME=${RUN_NAME:-qwen25vl3b_active_graph_sft_aligned_${PIPELINE_STAMP}}
export ADAPTER=${ADAPTER:-$PROJECT/outputs/checkpoints/$RUN_NAME}
export MERGED_MODEL=${MERGED_MODEL:-/scratch/punim0614/lifuzhang/models/${RUN_NAME}-merged}
cd "$PROJECT"
"$PYTHON" scripts/merge_qwen25_active_graph_sft.py --base-model "$BASE_MODEL" --adapter "$ADAPTER" --output "$MERGED_MODEL" --force
ln -sfn "$MERGED_MODEL" /scratch/punim0614/lifuzhang/active_graph_rl_workspace/models/sft_merged
printf 'MERGED_MODEL=%s\n' "$MERGED_MODEL"
