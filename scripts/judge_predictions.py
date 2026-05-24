#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.judge import add_judge_summary, judge_prediction_row, load_dotenv
from scripts.evaluate import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge an existing predictions.jsonl with DeepSeek.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl from evaluate.py")
    parser.add_argument("--summary", default=None, help="Optional summary.json path to update/write")
    parser.add_argument("--output", default=None, help="Optional output JSONL path. Defaults to overwriting predictions.")
    parser.add_argument("--baseline", choices=["agent", "agentic_summary"], default="agent")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Keep existing judge entries.")
    args = parser.parse_args()

    load_dotenv()
    pred_path = Path(args.predictions)
    rows = read_jsonl(pred_path)
    for idx, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id", idx - 1)
        if args.skip_existing and isinstance(row.get("judge"), dict):
            print(f"skip_existing {idx}/{len(rows)} sample_id={sample_id}", flush=True)
            continue
        print(f"judging {idx}/{len(rows)} sample_id={sample_id}", flush=True)
        try:
            row["judge"] = judge_prediction_row(
                row,
                model=args.model,
                base_url=args.base_url,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
            )
            row.pop("judge_error", None)
        except Exception as exc:
            row["judge_error"] = str(exc)
            print(f"judge_error sample_id={sample_id}: {exc}", flush=True)

    out_path = Path(args.output) if args.output else pred_path
    write_jsonl(out_path, rows)
    summary = summarize(rows, baseline=args.baseline)
    add_judge_summary(summary, rows)
    summary_path = Path(args.summary) if args.summary else out_path.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"predictions": str(out_path), "summary": str(summary_path), "summary_json": summary}, ensure_ascii=False, indent=2))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
