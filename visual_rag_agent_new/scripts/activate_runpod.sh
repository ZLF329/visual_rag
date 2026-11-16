#!/usr/bin/env bash
# Source this after attaching the preserved regional network volume at /workspace.
export WORK_ROOT=${WORK_ROOT:-/workspace}
export VISUAL_RAG_AGENT_PATH=$WORK_ROOT/code/visual_rag_agent_new
export PYTHONPATH=$WORK_ROOT/code/verl-agent:$VISUAL_RAG_AGENT_PATH${PYTHONPATH:+:$PYTHONPATH}
export PATH=$WORK_ROOT/envs/rl/bin:$PATH
export HF_HOME=$WORK_ROOT/cache/huggingface
export PIP_CACHE_DIR=$WORK_ROOT/cache/pip
export UV_CACHE_DIR=$WORK_ROOT/cache/uv
export UV_PYTHON_INSTALL_DIR=$WORK_ROOT/python
export XDG_CACHE_HOME=$WORK_ROOT/cache
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Existing index and adapter metadata use the original root. Preserve that path
# without rewriting data or overwriting an existing directory.
if [ ! -e /root/rivermind-data ] && [ ! -L /root/rivermind-data ]; then
  ln -s "$WORK_ROOT" /root/rivermind-data
elif [ "$(readlink -f /root/rivermind-data)" != "$(readlink -f "$WORK_ROOT")" ]; then
  printf 'Existing /root/rivermind-data points elsewhere; check before loading the data.\n' >&2
  return 2 2>/dev/null || exit 2
fi
