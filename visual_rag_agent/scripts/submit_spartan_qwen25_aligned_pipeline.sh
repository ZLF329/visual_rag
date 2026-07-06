#!/usr/bin/env bash
set -euo pipefail
cd /scratch/punim0614/lifuzhang/visual_rag_agent
PIPELINE_STAMP=${PIPELINE_STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_NAME=${RUN_NAME:-qwen25vl3b_active_graph_sft_aligned_${PIPELINE_STAMP}}
RL_RUN_ID=${RL_RUN_ID:-active_graph_qwen25vl3b_rl_aligned_${PIPELINE_STAMP}}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
SAVE_FREQ=${SAVE_FREQ:-100}
ACTIVE_GRAPH_EVAL_JUDGE=${ACTIVE_GRAPH_EVAL_JUDGE:-deepseek}
COMMON_EXPORT="ALL,PIPELINE_STAMP=${PIPELINE_STAMP},RUN_NAME=${RUN_NAME},RL_RUN_ID=${RL_RUN_ID},TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS},SAVE_FREQ=${SAVE_FREQ},ACTIVE_GRAPH_EVAL_JUDGE=${ACTIVE_GRAPH_EVAL_JUDGE}"

echo "PIPELINE_STAMP=$PIPELINE_STAMP"
echo "RUN_NAME=$RUN_NAME"
echo "RL_RUN_ID=$RL_RUN_ID"
echo "COMMON_EXPORT=$COMMON_EXPORT"

SFT_JOB=$(sbatch --parsable --export="$COMMON_EXPORT" scripts/spartan_ag_qwen25_sft_aligned.slurm)
echo "SFT_JOB=$SFT_JOB"
MERGE_JOB=$(sbatch --parsable --dependency=afterok:$SFT_JOB --export="$COMMON_EXPORT" scripts/spartan_ag_qwen25_merge_aligned.slurm)
echo "MERGE_JOB=$MERGE_JOB"
SFT_EVAL_JOB=$(sbatch --parsable --dependency=afterok:$MERGE_JOB --export="$COMMON_EXPORT" scripts/spartan_ag_qwen25_sft_eval_aligned.slurm)
echo "SFT_EVAL_JOB=$SFT_EVAL_JOB"
RL_JOB=$(sbatch --parsable --dependency=afterok:$SFT_EVAL_JOB --export="$COMMON_EXPORT" scripts/spartan_ag_qwen25_rl_train_aligned.slurm)
echo "RL_JOB=$RL_JOB"
RL_EVAL_JOB=$(sbatch --parsable --dependency=afterok:$RL_JOB --export="$COMMON_EXPORT" scripts/spartan_ag_qwen25_rl_eval_aligned.slurm)
echo "RL_EVAL_JOB=$RL_EVAL_JOB"

echo "Monitor: squeue -u $USER"
echo "Logs: /scratch/punim0614/lifuzhang/slurm_logs"
