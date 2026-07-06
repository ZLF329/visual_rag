#!/usr/bin/env bash
#SBATCH --job-name=ag-rl-qwen25
#SBATCH --partition=gpu-h100
#SBATCH --account=punim0614
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=07:00:00
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
export WORKSPACE=/scratch/punim0614/lifuzhang/active_graph_rl_workspace
export VERL_AGENT_PATH=/scratch/punim0614/lifuzhang/verl-agent
export BASE_MODEL=/scratch/punim0614/lifuzhang/models/Qwen2.5-VL-3B-Instruct
export SFT_ADAPTER=/scratch/punim0614/lifuzhang/visual_rag_agent/outputs/checkpoints/active_graph_qwen25vl3b_sft_oldloss_1229_qwenzoom_r16a32_spartan
export MERGED_MODEL=/scratch/punim0614/lifuzhang/models/Qwen2.5-VL-3B-Instruct-activegraph-sft-oldloss-merged
export PYTHON="$CONDA_ENV_PREFIX/bin/python"

cd "$PROJECT"
python scripts/merge_qwen25_active_graph_sft.py   --base-model "$BASE_MODEL"   --adapter "$SFT_ADAPTER"   --output "$MERGED_MODEL"   --force

ln -sfn "$MERGED_MODEL" "$WORKSPACE/models/sft_merged"

cd "$WORKSPACE"
ray stop --force || true

export MODEL_PATH="$MERGED_MODEL"
export VISUAL_RAG_AGENT_PATH="$PROJECT"
export DATA_ROOT="$WORKSPACE/data/ra_grpo"
export SLIDEVQA_INDEX_PATH="$WORKSPACE/data/index"
export REWARD_FN_PATH="$WORKSPACE/code/rewards/visor_answer_reward.py"
export RETRIEVER_PATH=/scratch/punim0614/lifuzhang/models/Qwen3-VL-Embedding-8B
export ACTIVE_GRAPH_RL_ENV_FILE="$WORKSPACE/.env"
export NGPUS_PER_NODE=4
export TRAIN_BATCH_SIZE=8
export VAL_BATCH_SIZE=16
export GROUP_SIZE=8
export PPO_MINI_BATCH_SIZE=16
export MAX_STEPS=${MAX_STEPS:-15}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export SAVE_FREQ=${SAVE_FREQ:-${TOTAL_TRAINING_STEPS:-100}}
export TEST_FREQ=${TEST_FREQ:-50}
export TRAJECTORY_RESPONSE_LENGTH=16384
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
export FILTER_GROUPS_ENABLE=False
export FILTER_GROUPS_MAX_NUM_GEN_BATCHES=1

RUN_NAME=${RUN_NAME:-active_graph_rl_qwen25_oldloss_spartan}
ROLLOUT_DIR="$WORKSPACE/logs/rollouts/$RUN_NAME"
export TENSORBOARD_DIR="$WORKSPACE/logs/tensorboard/$RUN_NAME"
mkdir -p "$WORKSPACE/logs/rl_runs" "$ROLLOUT_DIR" "$TENSORBOARD_DIR" "$WORKSPACE/checkpoints/$RUN_NAME"

bash run/run_current_active_graph_rl.sh   trainer.experiment_name="$RUN_NAME"   trainer.total_training_steps="$TOTAL_TRAINING_STEPS"   trainer.save_freq="$SAVE_FREQ"   trainer.test_freq="$TEST_FREQ"   trainer.logger='["console","tensorboard"]'   trainer.rollout_data_dir="$ROLLOUT_DIR"   trainer.default_local_dir="$WORKSPACE/checkpoints/$RUN_NAME"   trainer.max_actor_ckpt_to_keep=1
