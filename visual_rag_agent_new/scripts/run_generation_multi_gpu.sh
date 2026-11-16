#!/bin/bash
# Teacher-trajectory generation over the whole train split with one vLLM replica per GPU (data
# parallel) and one generation process per replica on a disjoint index range. Resumable: re-running
# with the same OUT skips finished samples (per shard run dir OUT_gpu<i>).
# usage: OUT=full_v3 ./scripts/run_generation_multi_gpu.sh        (GPUS / DATASET / WORKERS / CONFIG overridable)
set -u
ROOT=${ROOT:-/root/rivermind-data}
CODE="$ROOT/code/visual_rag_agent_new"
OUT="${OUT:?set OUT}"
DATASET="${DATASET:-$ROOT/data/slidevqa/train.jsonl}"
CONFIG="${CONFIG:-config/teacher.yaml config/latest.yaml}"
WORKERS="${WORKERS:-16}"
GPUS="${GPUS:-$(nvidia-smi -L | wc -l)}"
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }

# 1. a clean GPU: no other vLLM / eval / generation
for pat in "vllm serve" "scripts/evaluate.py" "generate_sft_trajectories.py"; do
  pids=$(pgrep -f "$pat" | tr '\n' ' ')
  if [ -n "$pids" ]; then
    log "stopping '$pat' pids=$pids"
    for p in $pids; do kids=$(pgrep -P "$p" | tr '\n' ' '); kill "$p" 2>/dev/null; sleep 1; for k in $kids; do kill "$k" 2>/dev/null; done; done
  fi
done
for i in $(seq 36); do
  busy=0; for g in $(seq 0 $((GPUS-1))); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$m" -ge 2000 ] && busy=1; done
  [ $busy = 0 ] && break; sleep 5
done
[ $busy = 0 ] || { log "ABORT: GPUs not free"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 1; }

# 2. one teacher replica per GPU
for g in $(seq 0 $((GPUS-1))); do
  GPU=$g PORT=$((8000+g)) setsid nohup "$CODE/scripts/serve_teacher.sh" > "$ROOT/serve_gen_gpu$g.log" 2>&1 < /dev/null &
done
for g in $(seq 0 $((GPUS-1))); do
  up=0
  for i in $(seq 180); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$((8000+g))/v1/models)" = 200 ] && { up=1; break; }
    pgrep -f "vllm serve.*--port $((8000+g))" >/dev/null || { log "ABORT: replica $g exited, see serve_gen_gpu$g.log"; exit 1; }
    sleep 10
  done
  [ $up = 1 ] || { log "ABORT: replica $g not up after 30 min"; exit 1; }
  log "replica $g up on port $((8000+g))"
done

# 3. disjoint shards, one generation process per replica
TOTAL=$(wc -l < "$DATASET")
SHARD=$(( (TOTAL + GPUS - 1) / GPUS ))
log "dataset $TOTAL rows, $GPUS shard(s) of $SHARD"
cd "$CODE"
for g in $(seq 0 $((GPUS-1))); do
  cfg="$CONFIG"; [ $g -gt 0 ] && cfg="$CONFIG config/teacher_$((8000+g)).yaml"
  GPU=$g START=$((g*SHARD)) N=$SHARD OUT="${OUT}_gpu$g" WORKERS=$WORKERS CONFIG="$cfg" DATASET="$DATASET" \
    setsid nohup ./scripts/run_generation.sh > "$ROOT/outputs/new_sft/${OUT}_gpu$g.log" 2>&1 < /dev/null &
  log "shard $g: start=$((g*SHARD)) n=$SHARD -> outputs/new_sft/${OUT}_gpu$g (config: $cfg)"
done
sleep 30
while pgrep -f "generate_sft_trajectories.py" >/dev/null; do sleep 120; done
for g in $(seq 0 $((GPUS-1))); do
  f="$ROOT/outputs/new_sft/${OUT}_gpu$g/trajectories.jsonl"
  log "shard $g: $(wc -l < "$f" 2>/dev/null || echo 0) trajectories; last log: $(tail -1 "$ROOT/outputs/new_sft/${OUT}_gpu$g.log" | cut -c1-120)"
done
log "PIPELINE_DONE"
