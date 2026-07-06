#!/bin/bash
# Eval our 3B RL (supportjudge step_75, 69.16 SlideVQA) on MMLongBench-Doc 847-answerable, POOLED retrieval, GPU0, 0.5MP (train-aligned).
# Our env + Qwen3-VL-Embedding retriever + DeepSeek binary judge. Per-category breakdown after.
cd /root/autodl-tmp
HF=/root/autodl-tmp/active_graph_rl_workspace/checkpoints/qwen25vl3b_ag_ra_supportjudge_4card/global_step_75/actor/huggingface
RES=/root/autodl-tmp/eval_3b_mmlb847_sj_results.txt
DATASET=/root/autodl-tmp/mmlongbench_doc/eval/test_mmlb847.jsonl
INDEX=/root/autodl-tmp/mmlongbench_doc/index
PY=/root/autodl-tmp/envs/verl/bin/python
NAME=3b-mmlb847-sj
RUN=3b_mmlb847_sj
OUT=/root/autodl-tmp/visual_rag_agent/outputs/$RUN

: > "$RES"
echo "eval 3B(supportjudge step75) MMLongBench-Doc 847 POOLED 0.5MP GPU0 START $(date)" >> "$RES"

kill_vllm () { pkill -9 -f 'bin/vllm serve' 2>/dev/null; pkill -9 -f 'VLLM::EngineCore' 2>/dev/null; sleep 8; }

kill_vllm
echo "[$(date)] serve 3B" >> "$RES"
MODEL_PATH=$HF SERVED_NAME=$NAME GMEM=0.55 TP=1 PORT=8000 VLLM_DEVICES=0 \
  setsid nohup bash /root/autodl-tmp/start_vllm.sh > /root/autodl-tmp/vllm_${NAME}.log 2>&1 </dev/null &
UP=0
for i in $(seq 1 180); do
  curl -s localhost:8000/v1/models 2>/dev/null | grep -q "$NAME" && { UP=1; break; }
  sleep 10
done
[ "$UP" = 1 ] || { echo "VLLM_FAILED (vllm_${NAME}.log)" >> "$RES"; exit 1; }
echo "[$(date)] predicting 847 (pooled index, MIN_PIXELS=500000, RETRIEVE_K=5)" >> "$RES"

ACTIVE_GRAPH_MIN_PIXELS=500000 ACTIVE_GRAPH_RETRIEVE_K=5 RUN_ID=$RUN NUM=847 MODELNAME=$NAME CONC=32 PORT=8000 EVAL_GPU=0 \
  INDEX=$INDEX DATASET=$DATASET RETRIEVER=/root/autodl-tmp/models/Qwen3-VL-Embedding-8B \
  bash /root/autodl-tmp/autodl_run_eval_predict.sh > /root/autodl-tmp/predict_${NAME}.log 2>&1

P=$(ls -t $OUT/*/predictions.jsonl 2>/dev/null | head -1)
NP=$(wc -l < "$P" 2>/dev/null | tr -d ' ')
TERM=$(grep -aoE "terminated_by=[a-z_]+" /root/autodl-tmp/predict_${NAME}.log | sort | uniq -c | tr '\n' ' ')
echo "[$(date)] judging (preds=$NP path=$P)" >> "$RES"
kill_vllm

DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?set me} $PY /root/autodl-tmp/judge_batch.py "$P" > /root/autodl-tmp/judge_${NAME}.log 2>&1
JD=$(grep JUDGE_DONE /root/autodl-tmp/judge_${NAME}.log | tail -1)
OKN=$(echo "$JD" | grep -oE 'file_correct=[0-9]+' | grep -oE '[0-9]+')
ACC=$($PY -c "print(f'{100*$OKN/847:.2f}')" 2>/dev/null)
echo "OVERALL: acc=${ACC}% ($JD) preds=$NP | term: $TERM" >> "$RES"
echo "[$(date)] per-category breakdown" >> "$RES"
DATASET=$DATASET PREDS=$P $PY /root/autodl-tmp/score_mmlb_bycat.py >> "$RES" 2>&1
echo "ALL_EVAL_DONE $(date)" >> "$RES"
