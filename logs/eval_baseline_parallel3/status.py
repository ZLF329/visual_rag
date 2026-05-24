from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT = Path('/root/autodl-tmp/visual_rag_agent')
ROOT = PROJECT / 'outputs/eval_baseline_parallel3'
LOG_DIR = PROJECT / 'logs/eval_baseline_parallel3'
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
    if path is None or not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding='utf-8').splitlines() if line.strip())


def last_eval(path: Path) -> str:
    if not path.exists():
        return ''
    lines = [line for line in path.read_text(encoding='utf-8', errors='replace').splitlines() if line.startswith('[eval]')]
    return lines[-1] if lines else ''

print('baseline parallel shards:')
for name, expected in SHARDS:
    run = latest_child(ROOT / f'agentic_summary_{name}')
    pid_path = LOG_DIR / f'{name}.pid'
    pid = pid_path.read_text().strip() if pid_path.exists() else ''
    rows = count_jsonl(run / 'predictions.jsonl' if run else None)
    summary = bool(run and (run / 'summary.json').exists())
    print(f'  {name}: rows={rows}/{expected} running={pid_running(pid)} summary={summary} last="{last_eval(LOG_DIR / (name + ".log"))}"')
watch_pid = (LOG_DIR / 'watch_merge.pid').read_text().strip() if (LOG_DIR / 'watch_merge.pid').exists() else ''
print(f'watch_merge: pid={watch_pid} running={pid_running(watch_pid, "watch_merge.py")}')
watch_log = LOG_DIR / 'watch_merge.log'
if watch_log.exists():
    for line in watch_log.read_text(encoding='utf-8', errors='replace').splitlines()[-5:]:
        print('  watch:', line)
merged = latest_child(ROOT / 'merged_agentic_summary')
print(f'merged_run={merged}')
