#!/usr/bin/env bash
#SBATCH --job-name=ag4b-rl2
#SBATCH --partition=gpu-h100
#SBATCH --account=punim0614
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/punim0614/lifuzhang/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/punim0614/lifuzhang/slurm_logs/%x-%j.err
set -euo pipefail
module load Miniconda3/23.10.0-1
module load CUDA/12.8.0
export CACHE_ROOT=${CACHE_ROOT:-/scratch/punim0614/.cache}
export CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-$CACHE_ROOT/conda/envs/verl-vllm011}
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
export PYTHON="$CONDA_ENV_PREFIX/bin/python"
export PYTHONPATH="$VERL_AGENT_PATH:$PROJECT:${PYTHONPATH:-}"
if [[ -f "$WORKSPACE/.env" ]]; then set -a; source "$WORKSPACE/.env"; set +a; fi
export PIPELINE_STAMP=${PIPELINE_STAMP:-$(date +%Y%m%d_%H%M%S)}
export RL_RUN_ID=${RL_RUN_ID:-active_graph_qwen3vl4b_rl_2gpu_${PIPELINE_STAMP}}
export MERGED_MODEL=${MERGED_MODEL:-/scratch/punim0614/lifuzhang/models/qwen3vl4b_active_graph_sft_NEWSM_20260613-merged}
export MODEL_PATH="$MERGED_MODEL"
export DATA_ROOT=/scratch/punim0614/lifuzhang/visual_rag_agent/data/ra_grpo/active_graph_rl_800single_400multi_20260614
export TRAIN_DATA="$DATA_ROOT/train.parquet"
export VAL_DATA="$DATA_ROOT/val.parquet"
export VISUAL_RAG_AGENT_PATH="$PROJECT"
export SLIDEVQA_INDEX_PATH=/scratch/punim0614/lifuzhang/visual_rag_agent/data/indexes/active_graph_rl_800single_400multi_20260614
export REWARD_FN_PATH="$WORKSPACE/code/rewards/visor_answer_reward.py"
export RETRIEVER_PATH=/scratch/punim0614/lifuzhang/models/Qwen3-VL-Embedding-8B
export ACTIVE_GRAPH_RL_ENV_FILE="$WORKSPACE/.env"
export ACTIVE_GRAPH_RETRIEVE_K=5
export NGPUS_PER_NODE=2
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
export GROUP_SIZE=${GROUP_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
export MAX_STEPS=${MAX_STEPS:-15}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-75}
export SAVE_FREQ=${SAVE_FREQ:-75}
export TEST_FREQ=${TEST_FREQ:-1000000}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export TRAJECTORY_RESPONSE_LENGTH=${TRAJECTORY_RESPONSE_LENGTH:-16384}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}
export ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}
export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
export ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}
export ROLLOUT_LIMIT_IMAGES=${ROLLOUT_LIMIT_IMAGES:-2}
export FILTER_GROUPS_ENABLE=False
export FILTER_GROUPS_MAX_NUM_GEN_BATCHES=1
export ACTOR_CHECKPOINT_CONTENTS='[model,optimizer,extra]'
if [[ ! -e "$MODEL_PATH/config.json" ]]; then
  echo "merged 4B model missing - merging LoRA on node"
  "$PYTHON" /scratch/punim0614/lifuzhang/merge_4b_lora_spartan.py
fi
for p in "$MODEL_PATH/config.json" "$TRAIN_DATA" "$VAL_DATA" "$SLIDEVQA_INDEX_PATH/embeddings.npy" "$RETRIEVER_PATH/config.json"; do
  [[ -e "$p" ]] || { echo "missing $p" >&2; exit 2; }
done
cd "$WORKSPACE"
ray stop --force || true
ROLLOUT_DIR="$WORKSPACE/logs/rollouts/$RL_RUN_ID"
CKPT_DIR="$WORKSPACE/checkpoints/$RL_RUN_ID"
mkdir -p "$ROLLOUT_DIR" "$CKPT_DIR" "$WORKSPACE/logs/rl_runs"
bash run/run_current_active_graph_rl.sh \
  trainer.experiment_name="$RL_RUN_ID" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.logger='["console","tensorboard"]' \
  trainer.rollout_data_dir="$ROLLOUT_DIR" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.max_actor_ckpt_to_keep=2 \
  '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=["Qwen3VLTextDecoderLayer"]' \
  '+actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=["Qwen3VLTextDecoderLayer"]' \
  actor_rollout_ref.actor.optim.warmup_style=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.2 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.3
