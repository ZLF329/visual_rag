from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROJECT = Path('/root/autodl-tmp/visual_rag_agent')
MAIN_ROOT = PROJECT / 'outputs/eval_parallel3'
MAIN_LOG = PROJECT / 'logs/eval_parallel3'
BASE_ROOT = PROJECT / 'outputs/eval_baseline200/agentic_summary'
BASE_LOG = PROJECT / 'logs/eval_baseline200'
SHARDS = [('part0', 67), ('part1', 67), ('part2', 66)]


def pid_running(pid: str, marker: str = 'scripts/evaluate.py') -> bool:
    if not pid:
        return False
    r = subprocess.run(['ps', '-p', pid, '-o', 'args='], capture_output=True, text=True, check=False)
    return r.returncode == 0 and marker in r.stdout


def latest_child(root: Path) -> Path | None:
    if not root.exists():
        return None
    kids = [p for p in root.iterdir() if p.is_dir()]
    return max(kids, key=lambda p: p.stat().st_mtime) if kids else None


def count_jsonl(path: Path | None) -> int:
    if not path or not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding='utf-8').splitlines() if line.strip())


def last_eval_line(path: Path) -> str:
    if not path.exists():
        return ''
    lines = [line for line in path.read_text(encoding='utf-8', errors='replace').splitlines() if line.startswith('[eval]')]
    return lines[-1] if lines else ''

print('main shards:')
for name, expected in SHARDS:
    run = latest_child(MAIN_ROOT / f'main_agent_{name}')
    pid_path = MAIN_LOG / f'{name}.pid'
    pid = pid_path.read_text().strip() if pid_path.exists() else ''
    rows = count_jsonl(run / 'predictions.jsonl' if run else None)
    summary = bool(run and (run / 'summary.json').exists())
    print(f'  {name}: rows={rows}/{expected} running={pid_running(pid)} summary={summary} last="{last_eval_line(MAIN_LOG / (name + ".log"))}"')

watch_pid = (BASE_LOG / 'watch_then_baseline.pid').read_text().strip() if (BASE_LOG / 'watch_then_baseline.pid').exists() else ''
print(f'watcher: pid={watch_pid} running={pid_running(watch_pid, "watch_then_baseline.py")}')
watch_log = BASE_LOG / 'watch_then_baseline.log'
if watch_log.exists():
    lines = watch_log.read_text(encoding='utf-8', errors='replace').splitlines()[-4:]
    for line in lines:
        print('  watch:', line)

base_pid = (BASE_LOG / 'baseline.pid').read_text().strip() if (BASE_LOG / 'baseline.pid').exists() else ''
base_run = latest_child(BASE_ROOT)
base_rows = count_jsonl(base_run / 'predictions.jsonl' if base_run else None)
base_summary = bool(base_run and (base_run / 'summary.json').exists())
print(f'baseline: pid={base_pid} running={pid_running(base_pid)} rows={base_rows}/200 summary={base_summary} run={base_run}')
print(f'baseline last="{last_eval_line(BASE_LOG / "baseline_agentic_summary.log")}"')
