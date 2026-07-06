#!/usr/bin/env bash
#SBATCH --job-name=ag25-rlevalvc
#SBATCH --partition=gpu-a100-short
#SBATCH --account=punim0614
#SBATCH --qos=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:3
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
export PROJECT=/scratch/punim0614/lifuzhang/visual_rag_agent
export WORKSPACE=/scratch/punim0614/lifuzhang/active_graph_rl_workspace
export VERL_AGENT_PATH=/scratch/punim0614/lifuzhang/verl-agent
export PYTHON="$CONDA_ENV_PREFIX/bin/python"
export PYTHONPATH="$VERL_AGENT_PATH:$PROJECT:${PYTHONPATH:-}"
if [[ -f "$WORKSPACE/.env" ]]; then set -a; source "$WORKSPACE/.env"; set +a; fi
export PIPELINE_STAMP=${PIPELINE_STAMP:-$(date +%Y%m%d_%H%M%S)}
export RUN_NAME=${RUN_NAME:-qwen25vl3b_sft_NEWSM_manualcrop_VISION_spartan_${PIPELINE_STAMP}}
export RL_RUN_ID=${RL_RUN_ID:-active_graph_qwen25vl3b_rl_VISIONcrop_spartan_${PIPELINE_STAMP}}
export MERGED_MODEL=${MERGED_MODEL:-/scratch/punim0614/lifuzhang/models/${RUN_NAME}-merged}
export RL_STEP=${RL_STEP:-100}
export RL_CKPT_ACTOR=${RL_CKPT_ACTOR:-$WORKSPACE/checkpoints/$RL_RUN_ID/global_step_${RL_STEP}/actor}
export OUT_MODEL=${OUT_MODEL:-/scratch/punim0614/lifuzhang/models/${RL_RUN_ID}_step${RL_STEP}_hf}
export BASE_MODEL="$MERGED_MODEL"
export WORLD_SIZE=${WORLD_SIZE:-4}
export MAX_SHARD_GB=${MAX_SHARD_GB:-4}
for p in "$RL_CKPT_ACTOR" "$BASE_MODEL/config.json"; do [[ -e "$p" ]] || { echo "missing $p" >&2; exit 2; }; done
"$PYTHON" "$WORKSPACE/run/export_dtensor_fsdp_to_hf.py"
cd "$PROJECT"
export EVAL_RUN_ID=${EVAL_RUN_ID:-${RL_RUN_ID}_eval500_${PIPELINE_STAMP}}
export ACTIVE_GRAPH_EVAL_JUDGE=${ACTIVE_GRAPH_EVAL_JUDGE:-deepseek}
export ACTIVE_GRAPH_RETRIEVE_K=${ACTIVE_GRAPH_RETRIEVE_K:-5}
export ACTIVE_GRAPH_EVAL_HISTORY_WINDOW=${ACTIVE_GRAPH_EVAL_HISTORY_WINDOW:-2}
export ACTIVE_GRAPH_EVAL_MAX_ITERS=${ACTIVE_GRAPH_EVAL_MAX_ITERS:-15}
"$PYTHON" scripts/spartan_run_active_graph_eval_2gpu.py --model-path "$OUT_MODEL" --run-id "$EVAL_RUN_ID" --num-samples 500 --shards ${EVAL_SHARDS:-3} --judge "$ACTIVE_GRAPH_EVAL_JUDGE" --judge-model "${ACTIVE_GRAPH_EVAL_JUDGE_MODEL:-deepseek-v4-flash}"
