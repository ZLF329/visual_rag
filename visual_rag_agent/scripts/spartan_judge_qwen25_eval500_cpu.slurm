#!/usr/bin/env bash
#SBATCH --job-name=ag-judge-eval500
#SBATCH --partition=sapphire
#SBATCH --account=punim0614
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/punim0614/lifuzhang/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/punim0614/lifuzhang/slurm_logs/%x-%j.err

set -euo pipefail
module load Miniconda3/23.10.0-1
export CACHE_ROOT=${CACHE_ROOT:-/scratch/punim0614/.cache}
export CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-$CACHE_ROOT/conda/envs/verl-gspo}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PREFIX" || true
export PATH="$CONDA_ENV_PREFIX/bin:$PATH"
export PY="$CONDA_ENV_PREFIX/bin/python"
export PYTHONPATH=/scratch/punim0614/lifuzhang/visual_rag_agent:${PYTHONPATH:-}
export ACTIVE_GRAPH_RL_ENV_FILE=/scratch/punim0614/lifuzhang/active_graph_rl_workspace/.env
# Load API env without printing secrets.
set -a
source "$ACTIVE_GRAPH_RL_ENV_FILE"
set +a

cd /scratch/punim0614/lifuzhang/visual_rag_agent
PRED=outputs/active_graph_qwen25vl3b_sft_eval500_oldloss_spartan_retry2/merged/20260607_012505/predictions.jsonl
OUT=outputs/active_graph_qwen25vl3b_sft_eval500_oldloss_spartan_retry2/merged/20260607_012505/judged_predictions.deepseek_v4_flash.jsonl
SUMMARY=outputs/active_graph_qwen25vl3b_sft_eval500_oldloss_spartan_retry2/merged/20260607_012505/summary.deepseek_v4_flash.json

"$PY" scripts/judge_predictions.py \
  --predictions "$PRED" \
  --output "$OUT" \
  --summary "$SUMMARY" \
  --model deepseek-v4-flash \
  --timeout 60 \
  --max-retries 2 \
  --max-tokens 512 \
  --skip-existing
