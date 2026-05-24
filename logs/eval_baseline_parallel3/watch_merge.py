#!/usr/bin/env python
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path('/root/autodl-tmp/visual_rag_agent')
LOG_DIR = PROJECT / 'logs/eval_baseline_parallel3'
ROOT = PROJECT / 'outputs/eval_baseline_parallel3'
MERGED_ROOT = ROOT / 'merged_agentic_summary'
SHARDS = [('part0', 67), ('part1', 67), ('part2', 66)]
POLL_SECONDS = 30


def log(msg: str) -> None:
    print(time.strftime('%Y-%m-%d %H:%M:%S'), msg, flush=True)


def pid_running(pid: str) -> bool:
    if not pid:
        return False
    r = subprocess.run(['ps', '-p', pid, '-o', 'args='], capture_output=True, text=True, check=False)
    return r.returncode == 0 and 'scripts/evaluate.py' in r.stdout


def latest_child(root: Path) -> Path | None:
    if not root.exists():
        return None
    kids = [p for p in root.iterdir() if p.is_dir()]
    return max(kids, key=lambda p: p.stat().st_mtime) if kids else None


def rows(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding='utf-8').splitlines() if line.strip())


def status() -> list[dict]:
    out = []
    for name, expected in SHARDS:
        pid_path = LOG_DIR / f'{name}.pid'
        pid = pid_path.read_text().strip() if pid_path.exists() else ''
        run = latest_child(ROOT / f'agentic_summary_{name}')
        out.append({
            'name': name,
            'expected': expected,
            'pid': pid,
            'running': pid_running(pid),
            'run': str(run) if run else None,
            'rows': rows(run / 'predictions.jsonl' if run else None),
            'summary': bool(run and (run / 'summary.json').exists()),
        })
    return out


def main() -> int:
    log('baseline parallel merge watcher started')
    while any(item['running'] for item in status()):
        log('running: ' + ', '.join(f"{x['name']} rows={x['rows']}/{x['expected']} running={x['running']}" for x in status()))
        time.sleep(POLL_SECONDS)
    problems = []
    roots = []
    for item in status():
        if item['rows'] != item['expected']:
            problems.append(f"{item['name']} rows={item['rows']} expected={item['expected']}")
        if not item['summary']:
            problems.append(f"{item['name']} missing summary.json")
        if item['run']:
            roots.append(item['run'])
    if problems:
        log('not merging: ' + '; '.join(problems))
        return 1
    cmd = [sys.executable, 'scripts/merge_eval_shards.py', '--roots', *roots, '--output', str(MERGED_ROOT), '--baseline', 'agentic_summary']
    log('merging: ' + ' '.join(cmd))
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    rc = subprocess.run(cmd, cwd=PROJECT, env=env, check=False).returncode
    log(f'merge finished rc={rc}')
    return rc

if __name__ == '__main__':
    raise SystemExit(main())
