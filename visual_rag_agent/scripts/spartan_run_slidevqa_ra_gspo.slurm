#!/usr/bin/env bash
#SBATCH --job-name=slidevqa-ra-gspo
#SBATCH --nodes=1
#SBATCH --partition=gpu-h100
#SBATCH --account=punim0614
#SBATCH --qos=publicgpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Set these on Spartan before sbatch if your paths differ.
export VISUAL_RAG_AGENT_PATH=${VISUAL_RAG_AGENT_PATH:-/scratch/punim0614/lifuzhang/visual_rag_agent}
export VERL_AGENT_PATH=${VERL_AGENT_PATH:-/scratch/punim0614/lifuzhang/verl-agent}
export MODEL_PATH=${MODEL_PATH:-/scratch/punim0614/lifuzhang/models/Qwen3-VL-4B-Instruct}
export TRAIN_DATA=${TRAIN_DATA:-${VISUAL_RAG_AGENT_PATH}/data/ra_grpo/slidevqa/train.parquet}
export VAL_DATA=${VAL_DATA:-${VISUAL_RAG_AGENT_PATH}/data/ra_grpo/slidevqa/val.parquet}
export SLIDEVQA_INDEX_PATH=${SLIDEVQA_INDEX_PATH:-${VISUAL_RAG_AGENT_PATH}/data/indexes/slidevqa_train_balanced_2000_plus_ra800_full_decks}
export REWARD_FN_PATH=${REWARD_FN_PATH:-${VISUAL_RAG_AGENT_PATH}/rewards/visor_answer_reward.py}
export RETRIEVER_PATH=${RETRIEVER_PATH:-/scratch/punim0614/lifuzhang/models/Qwen3-VL-Embedding-8B}

# Keep all runtime caches off $HOME on Spartan.
export CACHE_ROOT=${CACHE_ROOT:-/scratch/punim0614/.cache}
mkdir -p "$CACHE_ROOT"/{huggingface,torch,triton,ray,tmp,wandb}
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export RAY_TMPDIR="$CACHE_ROOT/ray"
export TMPDIR="$CACHE_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
export NNODES=${NNODES:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
export GROUP_SIZE=${GROUP_SIZE:-8}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-50}

mkdir -p logs
module load Miniconda3/23.10.0-1
module load CUDA/12.4.1
export CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-$CACHE_ROOT/conda/envs/verl-gspo}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PREFIX"
cd "$VERL_AGENT_PATH"

# Freeze policy in run_slidevqa_ra.sh:
#   freeze_vision_tower=True
#   freeze_multi_modal_projector=True
#   freeze_language_model=False
bash examples/gigpo_trainer/run_slidevqa_ra.sh vllm "$@"
