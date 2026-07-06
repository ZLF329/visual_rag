#!/usr/bin/env python
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path('/root/autodl-tmp/visual_rag_agent')
RUN_ID = os.environ.get('ACTIVE_GRAPH_EVAL_RUN_ID', f'active_graph_rl_step100_eval500_{time.strftime("%Y%m%d_%H%M%S")}')
EVAL_ROOT = PROJECT / 'outputs' / RUN_ID
LOG_DIR = PROJECT / 'logs' / RUN_ID
CONFIG = PROJECT / 'config' / 'active_graph_rl_step100_eval500.yaml'
DATASET = PROJECT / 'data/corpora/slidevqa/test.jsonl'

shards = [
    ('shard0', 0, 167, '0'),
    ('shard1', 167, 167, '1'),
    ('shard2', 334, 166, '2'),
]
EVAL_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

base_env = os.environ.copy()
base_env.update({
    'PYTHONPATH': str(PROJECT),
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'HF_HUB_DISABLE_PROGRESS_BARS': '1',
    'TRANSFORMERS_VERBOSITY': 'error',
})

procs = []
for name, start, count, gpu in shards:
    out = EVAL_ROOT / name
    env = base_env.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu
    log_path = LOG_DIR / f'{name}.log'
    cmd = [
        '/root/miniconda3/bin/python', 'scripts/evaluate.py',
        '--dataset-file', str(DATASET),
        '--config', str(CONFIG),
        '--output', str(out),
        '--start-index', str(start),
        '--num-samples', str(count),
        '--baseline', 'agent',
        '--judge', 'none',
    ]
    with log_path.open('ab') as log:
        p = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (LOG_DIR / f'{name}.pid').write_text(str(p.pid), encoding='utf-8')
    procs.append((name, p, out, log_path, gpu, start, count))
    print(json.dumps({'event': 'launched', 'name': name, 'pid': p.pid, 'gpu': gpu, 'start': start, 'count': count, 'log': str(log_path), 'output': str(out)}), flush=True)

failed = []
for name, p, out, log_path, gpu, start, count in procs:
    rc = p.wait()
    print(json.dumps({'event': 'finished', 'name': name, 'pid': p.pid, 'returncode': rc, 'log': str(log_path)}), flush=True)
    if rc != 0:
        failed.append((name, rc, str(log_path)))

if failed:
    print(json.dumps({'event': 'failed', 'failed': failed}, ensure_ascii=False), flush=True)
    raise SystemExit(1)

merge_cmd = [
    '/root/miniconda3/bin/python', 'scripts/merge_eval_shards.py',
    '--roots', *(str(item[2]) for item in procs),
    '--output', str(EVAL_ROOT / 'merged'),
    '--baseline', 'agent',
]
print(json.dumps({'event': 'merge_start', 'cmd': merge_cmd}, ensure_ascii=False), flush=True)
subprocess.run(merge_cmd, cwd=PROJECT, env=base_env, check=True)
print(json.dumps({'event': 'all_done', 'eval_root': str(EVAL_ROOT), 'log_dir': str(LOG_DIR)}, ensure_ascii=False), flush=True)
