#!/usr/bin/env python
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path('/root/autodl-tmp/visual_rag_agent')
LOG_DIR = PROJECT / 'logs/eval_baseline200'
MAIN_LOG_DIR = PROJECT / 'logs/eval_parallel3'
MAIN_ROOT = PROJECT / 'outputs/eval_parallel3'
BASELINE_ROOT = PROJECT / 'outputs/eval_baseline200/agentic_summary'
MERGED_ROOT = PROJECT / 'outputs/eval_parallel3/merged_main_agent'
DATASET = PROJECT / 'data/corpora/slidevqa/test.jsonl'
CONFIG = PROJECT / 'config/eval_flash.yaml'
INDEX = PROJECT / 'data/indexes/slidevqa'
SHARDS = [
    ('part0', 67),
    ('part1', 67),
    ('part2', 66),
]
POLL_SECONDS = 30


def log(message: str) -> None:
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def pid_running(pid: str) -> bool:
    if not pid:
        return False
    result = subprocess.run(['ps', '-p', pid, '-o', 'args='], capture_output=True, text=True, check=False)
    return result.returncode == 0 and 'scripts/evaluate.py' in result.stdout


def shard_pid(name: str) -> str:
    path = MAIN_LOG_DIR / f'{name}.pid'
    return path.read_text().strip() if path.exists() else ''


def latest_child(root: Path) -> Path | None:
    if not root.exists():
        return None
    children = [p for p in root.iterdir() if p.is_dir()]
    return max(children, key=lambda p: p.stat().st_mtime) if children else None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def shard_status() -> list[dict]:
    status = []
    for name, expected in SHARDS:
        root = MAIN_ROOT / f'main_agent_{name}'
        run = latest_child(root)
        predictions = run / 'predictions.jsonl' if run else None
        summary = run / 'summary.json' if run else None
        rows = read_jsonl(predictions) if predictions and predictions.exists() else []
        status.append({
            'name': name,
            'expected': expected,
            'pid': shard_pid(name),
            'running': pid_running(shard_pid(name)),
            'run': str(run) if run else None,
            'rows': len(rows),
            'summary': bool(summary and summary.exists()),
        })
    return status


def all_main_done() -> bool:
    return not any(item['running'] for item in shard_status())


def assert_main_complete() -> list[str]:
    roots = []
    problems = []
    for item in shard_status():
        if not item['run']:
            problems.append(f"{item['name']}: no run dir")
            continue
        if item['rows'] != item['expected']:
            problems.append(f"{item['name']}: rows={item['rows']} expected={item['expected']}")
        if not item['summary']:
            problems.append(f"{item['name']}: missing summary.json")
        roots.append(item['run'])
    if problems:
        raise RuntimeError('; '.join(problems))
    return roots


def run_checked(cmd: list[str], log_path: Path | None = None) -> int:
    log('running: ' + ' '.join(cmd))
    env = os.environ.copy()
    env.update({
        'PYTHONPATH': '.',
        'OMP_NUM_THREADS': '6',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
        'HF_HUB_DISABLE_PROGRESS_BARS': '1',
        'TRANSFORMERS_VERBOSITY': 'error',
        'HF_ENDPOINT': 'https://hf-mirror.com',
    })
    if log_path is None:
        return subprocess.run(cmd, cwd=PROJECT, env=env, check=False).returncode
    with log_path.open('w', encoding='utf-8') as f:
        proc = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=f, stderr=subprocess.STDOUT)
        (LOG_DIR / 'baseline.pid').write_text(str(proc.pid), encoding='utf-8')
        return proc.wait()


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log('watcher started')
    while not all_main_done():
        compact = ', '.join(
            f"{item['name']} rows={item['rows']}/{item['expected']} running={item['running']}"
            for item in shard_status()
        )
        log('main eval still running: ' + compact)
        time.sleep(POLL_SECONDS)

    log('main eval processes exited')
    roots = assert_main_complete()
    log('main eval complete: ' + ', '.join(roots))

    merge_cmd = [
        sys.executable, 'scripts/merge_eval_shards.py',
        '--roots', *roots,
        '--output', str(MERGED_ROOT),
        '--baseline', 'agent',
    ]
    merge_rc = run_checked(merge_cmd)
    if merge_rc != 0:
        log(f'merge failed rc={merge_rc}')
        return merge_rc

    baseline_cmd = [
        sys.executable, 'scripts/evaluate.py',
        '--baseline', 'agentic_summary',
        '--dataset-file', str(DATASET),
        '--start-index', '0',
        '--num-samples', '200',
        '--config', str(CONFIG),
        '--index', str(INDEX),
        '--output', str(BASELINE_ROOT),
        '--judge', 'deepseek',
        '--judge-model', os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-flash'),
    ]
    log('starting baseline')
    rc = run_checked(baseline_cmd, LOG_DIR / 'baseline_agentic_summary.log')
    log(f'baseline finished rc={rc}')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
