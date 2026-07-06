#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.judge import JudgeError, judge_prediction_row, load_dotenv


def now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()


def read_complete_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()
    # If writer is in the middle of a JSON row, skip the incomplete tail.
    if text and not text.endswith('\n') and lines:
        lines = lines[:-1]
    return [line for line in lines if line.strip()]


def summarize(judged_path: Path, summary_path: Path) -> dict[str, Any]:
    total = judged = correct = errors = 0
    by_source: dict[str, dict[str, int]] = {}
    if judged_path.exists():
        with judged_path.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                src = str(row.get('_judge_source') or 'unknown')
                item = by_source.setdefault(src, {'total': 0, 'judged': 0, 'correct': 0, 'errors': 0})
                item['total'] += 1
                if isinstance(row.get('judge'), dict):
                    judged += 1
                    item['judged'] += 1
                    if row['judge'].get('correct') is True:
                        correct += 1
                        item['correct'] += 1
                if row.get('judge_error'):
                    errors += 1
                    item['errors'] += 1
    summary = {
        'updated_at': now(),
        'total_rows_seen_by_judge': total,
        'judged': judged,
        'correct': correct,
        'errors': errors,
        'accuracy': (correct / judged) if judged else None,
        'by_source': by_source,
    }
    write_json(summary_path, summary)
    return summary


def run_once(args: argparse.Namespace, state: dict[str, Any], judged_path: Path, summary_path: Path) -> int:
    sources = sorted(args.output_root.glob(args.pattern))
    processed_this_round = 0
    for src in sources:
        src_key = str(src.resolve())
        lines = read_complete_lines(src)
        start = int(state.get('sources', {}).get(src_key, 0))
        if start > len(lines):
            start = 0
        new_lines = lines[start:]
        if not new_lines:
            continue
        print(f'[{now()}] source={src} complete_lines={len(lines)} new={len(new_lines)} start={start}', flush=True)
        for offset, line in enumerate(new_lines, start=start + 1):
            try:
                row = json.loads(line)
            except Exception as exc:
                out = {'_judge_source': src_key, '_judge_source_line': offset, 'judge_error': f'invalid prediction JSON: {exc}'}
                append_jsonl(judged_path, out)
            else:
                out = dict(row)
                out['_judge_source'] = src_key
                out['_judge_source_line'] = offset
                out['_judged_at'] = now()
                if isinstance(out.get('judge'), dict) or out.get('judge_error'):
                    pass
                else:
                    try:
                        out['judge'] = judge_prediction_row(
                            out,
                            model=args.model,
                            base_url=args.base_url,
                            timeout=args.timeout,
                            max_retries=args.max_retries,
                            max_tokens=args.max_tokens,
                        )
                    except Exception as exc:
                        out['judge_error'] = str(exc)
                        print(f'[{now()}] judge_error source={src.name} line={offset} sample_id={out.get("sample_id")}: {exc}', flush=True)
                append_jsonl(judged_path, out)
            state.setdefault('sources', {})[src_key] = offset
            state['updated_at'] = now()
            write_json(args.state_path, state)
            processed_this_round += 1
            if args.max_new and processed_this_round >= args.max_new:
                summarize(judged_path, summary_path)
                return processed_this_round
    summary = summarize(judged_path, summary_path)
    if processed_this_round:
        print(f'[{now()}] judged_new={processed_this_round} total_judged={summary["judged"]} correct={summary["correct"]} errors={summary["errors"]} accuracy={summary["accuracy"]}', flush=True)
    else:
        print(f'[{now()}] no new complete prediction rows', flush=True)
    return processed_this_round


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--pattern', default='shard*/20*/predictions.jsonl')
    parser.add_argument('--judge-dir', type=Path, default=None)
    parser.add_argument('--state-path', type=Path, default=None)
    parser.add_argument('--model', default='deepseek-v4-flash')
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--interval', type=float, default=300.0)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--max-new', type=int, default=0)
    args = parser.parse_args()

    load_dotenv(ROOT / '.env')
    args.output_root = args.output_root.resolve()
    judge_dir = (args.judge_dir or (args.output_root / 'judge_incremental')).resolve()
    judge_dir.mkdir(parents=True, exist_ok=True)
    args.state_path = (args.state_path or (judge_dir / 'judge_state.json')).resolve()
    judged_path = judge_dir / 'predictions_judged.jsonl'
    summary_path = judge_dir / 'judge_summary.json'
    state = load_json(args.state_path, {'sources': {}, 'created_at': now()})

    print(f'[{now()}] incremental judge started', flush=True)
    print(f'output_root={args.output_root}', flush=True)
    print(f'judged_path={judged_path}', flush=True)
    print(f'summary_path={summary_path}', flush=True)
    print(f'state_path={args.state_path}', flush=True)
    print(f'model={args.model} interval={args.interval}s pattern={args.pattern}', flush=True)

    while True:
        try:
            run_once(args, state, judged_path, summary_path)
        except Exception as exc:
            print(f'[{now()}] loop_error: {type(exc).__name__}: {exc}', flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
