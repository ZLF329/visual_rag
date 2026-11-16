#!/bin/bash
# Teacher trajectories over a prepared SlideVQA jsonl (scripts/prepare_slidevqa.py), flat corpus.
# Writes to outputs/new_sft/$OUT; re-running with the same OUT resumes (finished samples are skipped).
# Usage: DATASET=.../smoke30.jsonl OUT=smoke ./scripts/run_generation.sh     (default: all of train)
# Blackwell box: CONFIG="config/teacher.yaml config/blackwell.yaml" ./scripts/run_generation.sh
set -e
ROOT=${ROOT:-/root/rivermind-data}
PROJECT="$ROOT/code/visual_rag_agent_new"
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
export TEACHER_API_KEY=${TEACHER_API_KEY:-dummy}
set -a; source "$ROOT/.secrets.env"; set +a          # DEEPSEEK_API_KEY for the judge
export CUDA_VISIBLE_DEVICES=${GPU:-0}                 # the retriever's GPU; pair it with the vLLM replica there
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}          # the container's CPU quota is 22 cores (nproc says 100)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}   # the ColQwen flash-attn3 Hub kernel
DATASET="${DATASET:-$ROOT/data/slidevqa/train.jsonl}"
N="${N:-100000}"; START="${START:-0}"; WORKERS="${WORKERS:-16}"; OUT="${OUT:-full}"
CONFIG="${CONFIG:-config/teacher.yaml}"
exec "$ROOT/envs/vllm/bin/python" scripts/generate_sft_trajectories.py \
  --dataset-file "$DATASET" \
  --config $CONFIG \
  --run-dir "$ROOT/outputs/new_sft/$OUT" \
  --start-index "$START" --num-samples "$N" --workers "$WORKERS" \
  --judge deepseek --judge-model deepseek-v4-flash --require-judge-correct --require-all-reference-pages \
  --stop-on-api-error
