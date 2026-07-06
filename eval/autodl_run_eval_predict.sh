#!/bin/bash
set -uo pipefail
PY=/root/autodl-tmp/envs/verl/bin/python
PROJECT=/root/autodl-tmp/visual_rag_agent
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export HF_HOME=/root/autodl-tmp/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export ACTIVE_GRAPH_RETRIEVE_K=5
export ACTIVE_GRAPH_EVAL_ANSWER_EVIDENCE=${ACTIVE_GRAPH_EVAL_ANSWER_EVIDENCE:-0}  # 0 = non-verify (root accept -> answer, copy committed). verify=DEAD END
export ACTIVE_GRAPH_EVAL_ANSWER_NOIMG=${ACTIVE_GRAPH_EVAL_ANSWER_NOIMG:-0}        # verify-from-graph (off)
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy-vllm-key}
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:-${DEEPSEEK_API_KEY}}
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0}
RUN_ID=${RUN_ID:-sft_verify_vllm}
NUM=${NUM:-500}
START=${START:-0}
CONC=${CONC:-16}
PORT=${PORT:-8000}
MODELNAME=${MODELNAME:-qwen25vl3b-verify}
INDEX=${INDEX:-/root/autodl-tmp/active_graph_rl_workspace/data/indexes/slidevqa_test_main}
RETRIEVER=${RETRIEVER:-/root/autodl-tmp/models/Qwen3-VL-Embedding-8B}
DATASET=${DATASET:-$PROJECT/data/corpora/slidevqa/test.jsonl}
OUT=$PROJECT/outputs/$RUN_ID
CFG=$PROJECT/config/$RUN_ID.yaml
mkdir -p "$OUT" "$PROJECT/config"
[[ -f "$DATASET" ]] || { echo "MISSING DATASET $DATASET"; exit 2; }
cat > "$CFG" <<YAML
models:
  vlm:
    provider: openai_compatible
    name: $MODELNAME
    api_base_url: http://localhost:$PORT/v1
    api_key_env: OPENAI_API_KEY
    max_tokens: 1024
    temperature: 0.0
    prompt_mode: system_in_user
  retriever:
    name: $RETRIEVER
    index_path: $INDEX
agent:
  top_k: 1
  max_iters: 15
  partial_memory_capacity: 2
image_budget:
  yes_pixels: 400000
  partial_pixels: 400000
dataset:
  name: slidevqa
  split: test
  num_samples: $NUM
runtime:
  output_dir: $OUT
  device: cuda
  dtype: bfloat16
  attn_implementation: flash_attention_2
YAML
cd "$PROJECT"
echo "[vllm-eval] RUN_ID=$RUN_ID NUM=$NUM START=$START CONC=$CONC model=$MODELNAME"
$PY scripts/evaluate.py --dataset-file "$DATASET" --config "$CFG" --output "$OUT" \
  --baseline agent --start-index "$START" --num-samples "$NUM" \
  --judge none --judge-model deepseek-v4-flash --judge-max-tokens 1024 \
  --concurrency "$CONC" 2>&1
echo "VLLM_EVAL_DONE out=$OUT"
