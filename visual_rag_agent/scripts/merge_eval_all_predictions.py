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
    parser = argparse.ArgumentParser(description='Merge every predictions.jsonl under an eval output root, de-duplicating sample_id.')
    parser.add_argument('--root', required=True, help='Eval output root containing shard/resume run directories.')
    parser.add_argument('--output', default=None, help='Merged output root. Defaults to ROOT/merged_all.')
    parser.add_argument('--baseline', choices=['agent', 'agentic_summary'], default='agent')
    parser.add_argument('--expected-total', type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    root = Path(args.root)
    output_root = Path(args.output) if args.output else root / 'merged_all'
    rows_by_id: dict[str, tuple[float, int, dict[str, Any], str]] = {}
    used_files: list[str] = []
    order = 0
    for pred in sorted(root.glob('**/predictions.jsonl')):
        # Skip previous merged outputs under this same root.
        rel_parts = pred.relative_to(root).parts
        if rel_parts and rel_parts[0].startswith('merged'):
            continue
        used_files.append(str(pred))
        mtime = pred.stat().st_mtime
        for line in pred.read_text(encoding='utf-8', errors='replace').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get('sample_id'))
            if sid == 'None':
                sid = f'__missing_{order}'
            prev = rows_by_id.get(sid)
            candidate = (mtime, order, row, str(pred))
            if prev is None or candidate[:2] >= prev[:2]:
                rows_by_id[sid] = candidate
            order += 1

    rows = [item[2] for item in rows_by_id.values()]
    rows.sort(key=lambda row: int(row.get('sample_id', 10**9)) if str(row.get('sample_id')).isdigit() else 10**9)
    out_dir = output_root / time.strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / 'predictions.jsonl', rows)
    summary = summarize(rows, baseline=args.baseline)
    add_judge_summary(summary, rows)
    summary['merged_from_files'] = used_files
    summary['unique_sample_ids'] = len(rows_by_id)
    if args.expected_total is not None:
        target = {str(i) for i in range(args.expected_total)}
        have = set(rows_by_id)
        missing = sorted(int(i) for i in target - have)
        summary['expected_total'] = args.expected_total
        summary['missing_count'] = len(missing)
        summary['missing_sample_ids'] = missing
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out_dir), 'num_rows': len(rows), 'summary': summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
