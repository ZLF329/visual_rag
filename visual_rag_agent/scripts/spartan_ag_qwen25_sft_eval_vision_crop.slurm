#!/usr/bin/env bash
#SBATCH --job-name=ag25-evalvc
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
export PIPELINE_STAMP=${PIPELINE_STAMP:-$(date +%Y%m%d_%H%M%S)}
export RUN_NAME=${RUN_NAME:-qwen25vl3b_active_graph_sft_aligned_${PIPELINE_STAMP}}
export MERGED_MODEL=${MERGED_MODEL:-/scratch/punim0614/lifuzhang/models/${RUN_NAME}-merged}
export EVAL_RUN_ID=${EVAL_RUN_ID:-${RUN_NAME}_sft_eval500_${PIPELINE_STAMP}}
export ACTIVE_GRAPH_EVAL_JUDGE=${ACTIVE_GRAPH_EVAL_JUDGE:-deepseek}
export ACTIVE_GRAPH_RETRIEVE_K=${ACTIVE_GRAPH_RETRIEVE_K:-5}
export ACTIVE_GRAPH_EVAL_HISTORY_WINDOW=${ACTIVE_GRAPH_EVAL_HISTORY_WINDOW:-2}
if [[ -f /scratch/punim0614/lifuzhang/active_graph_rl_workspace/.env ]]; then set -a; source /scratch/punim0614/lifuzhang/active_graph_rl_workspace/.env; set +a; fi
cd "$PROJECT"
"$PYTHON" scripts/spartan_run_active_graph_eval_2gpu.py --model-path "$MERGED_MODEL" --run-id "$EVAL_RUN_ID" --num-samples 500 --shards 2 --judge "$ACTIVE_GRAPH_EVAL_JUDGE" --judge-model "${ACTIVE_GRAPH_EVAL_JUDGE_MODEL:-deepseek-v4-flash}"
