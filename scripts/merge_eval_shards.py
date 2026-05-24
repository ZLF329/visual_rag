#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.evaluate import summarize, write_jsonl
from src.judge import add_judge_summary, load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description='Merge sharded evaluate.py outputs.')
    parser.add_argument('--roots', nargs='+', required=True, help='Shard output roots or timestamped run dirs.')
    parser.add_argument('--output', required=True, help='Merged output directory root.')
    parser.add_argument('--baseline', choices=['agent', 'agentic_summary'], default='agent')
    args = parser.parse_args()

    load_dotenv()
    rows: list[dict[str, Any]] = []
    used_runs: list[str] = []
    for root_raw in args.roots:
        root = Path(root_raw)
        run = root if (root / 'predictions.jsonl').exists() else latest_child(root)
        if run is None or not (run / 'predictions.jsonl').exists():
            raise FileNotFoundError(f'No predictions.jsonl found under {root}')
        used_runs.append(str(run))
        rows.extend(read_jsonl(run / 'predictions.jsonl'))

    rows.sort(key=lambda row: int(row.get('sample_id', 10**9)))
    out_dir = Path(args.output) / time.strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / 'predictions.jsonl', rows)
    summary = summarize(rows, baseline=args.baseline)
    add_judge_summary(summary, rows)
    summary['merged_from'] = used_runs
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out_dir), 'num_rows': len(rows), 'summary': summary}, ensure_ascii=False, indent=2))


def latest_child(path: Path) -> Path | None:
    if not path.exists():
        return None
    children = [p for p in path.iterdir() if p.is_dir()]
    return max(children, key=lambda p: p.stat().st_mtime) if children else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


if __name__ == '__main__':
    main()
