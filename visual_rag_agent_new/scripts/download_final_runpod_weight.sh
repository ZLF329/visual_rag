#!/usr/bin/env bash
# Recover a slow public-weight download using HTTP ranges and the source SHA-256.
set -euo pipefail
export WORK_ROOT=${WORK_ROOT:-/workspace}
export HF_HOME=$WORK_ROOT/cache/huggingface
PREP=$WORK_ROOT/preparation
if [ -f "$PREP/retriever_weights_verified.json" ]; then
  exit 0
fi
trap 'rc=$?; printf "%s\n" "$rc" > "$PREP/retriever_aria2.exit"' EXIT
rm -f "$PREP/retriever_aria2.exit"
python3 - <<'PY'
from pathlib import Path
import os, signal
pid_file = Path(os.environ.get('WORK_ROOT', '/workspace')) / 'preparation/retriever_download.pid'
if pid_file.exists():
    pid = int(pid_file.read_text())
    command = Path(f'/proc/{pid}/cmdline')
    if command.exists() and b'scripts/download_runpod_retriever.py' in command.read_bytes():
        os.kill(pid, signal.SIGTERM)
        print('REPLACED_OWN_SLOW_DOWNLOADER', flush=True)
PY
aria2c --enable-color=false --console-log-level=warn --summary-interval=30 \
  --max-connection-per-server=8 --split=8 --min-split-size=16M \
  --file-allocation=none --continue=true --check-integrity=true \
  --checksum=sha-256=6b45c7afe391b4d9cc49f1ed3f6976f4a25ed40aa2165ed2ae118ff549355985 \
  --max-tries=10 --retry-wait=3 --timeout=30 --connect-timeout=15 \
  --allow-overwrite=true --auto-file-renaming=false \
  --dir="$WORK_ROOT/models/colqwen2.5-base" --out=model-00001-of-00002.safetensors \
  'https://huggingface.co/vidore/colqwen2.5-base/resolve/92908120384b7a2110c5beda3ab29cbdb2c08e49/model-00001-of-00002.safetensors?download=true'
"$WORK_ROOT/envs/rl/bin/python" "$WORK_ROOT/code/visual_rag_agent_new/scripts/download_runpod_retriever.py" --verify-only
