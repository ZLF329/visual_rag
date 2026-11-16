#!/bin/bash
# Evaluate the agent on a prepared SlideVQA split: flat corpus over that split's pages (SCORE's
# unified-corpus protocol), DeepSeek v4-flash judge. Writes to outputs/new_eval/$OUT; re-running with
# the same OUT resumes (finished samples are skipped).
# Usage: OUT=test_qwen38 ./scripts/run_eval.sh
#        SPLIT / DATASET / INDEX / CONFIG ("a.yaml b.yaml" overlays) / START / N / WORKERS / GPU are overridable.
#        Two GPUs: one shard per GPU (GPU=1 START=... with the replica on port 8001), then scripts/merge_eval_shards.py.
set -e
ROOT=${ROOT:-/root/rivermind-data}
PROJECT="$ROOT/code/visual_rag_agent_new"
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
export TEACHER_API_KEY=${TEACHER_API_KEY:-dummy}
set -a; source "$ROOT/.secrets.env"; set +a          # DEEPSEEK_API_KEY for the judge
export CUDA_VISIBLE_DEVICES=${GPU:-0}                # the retriever's query encoder
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}          # the container's CPU quota is 22 cores (nproc says 100)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}   # the ColQwen flash-attn3 Hub kernel
SPLIT="${SPLIT:-test}"
DATASET="${DATASET:-$ROOT/data/slidevqa/$SPLIT.jsonl}"
INDEX="${INDEX:-$ROOT/data/slidevqa/index/${SPLIT}_colqwen}"
CONFIG="${CONFIG:-config/teacher.yaml}"
START="${START:-0}"; N="${N:-100000}"; WORKERS="${WORKERS:-16}"; OUT="${OUT:?set OUT}"
exec "$ROOT/envs/vllm/bin/python" scripts/evaluate.py \
  --dataset-file "$DATASET" \
  --config $CONFIG \
  --index "$INDEX" \
  --run-dir "$ROOT/outputs/new_eval/$OUT" \
  --start-index "$START" --num-samples "$N" --concurrency "$WORKERS" \
  --judge deepseek --judge-model deepseek-v4-flash
