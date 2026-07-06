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
from src.judge import add_judge_summary, judge_prediction_row, load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch eval predictions and judge them in sidecar batches.")
    parser.add_argument("--root", required=True, type=Path, help="Eval output root containing shard directories.")
    parser.add_argument("--sidecar-dir", default=None, type=Path)
    parser.add_argument("--expected-total", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--judge-every-poll", action="store_true", help="Judge pending rows on every poll, capped by --batch-size.")
    parser.add_argument("--baseline", choices=["agent", "agentic_summary"], default="agent")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    load_dotenv()
    sidecar_dir = args.sidecar_dir or (args.root / "judge_sidecar")
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sidecar_dir / "judged_sidecar.jsonl"
    final_dir = sidecar_dir / "final"
    latest_dir = sidecar_dir / "latest"
    log_path = sidecar_dir / "watch_judge.log"

    while True:
        rows = load_latest_predictions(args.root)
        sidecar_rows = read_jsonl(sidecar_path) if sidecar_path.exists() else []
        sidecar_by_id, attempts_by_id = sidecar_index(sidecar_rows)
        eval_running = evaluate_running()
        completed = len(rows)
        batch = pending_rows(rows, sidecar_by_id, attempts_by_id, args.max_attempts)
        should_judge = (
            len(batch) >= args.batch_size
            or (args.judge_every_poll and bool(batch))
            or (completed >= args.expected_total and not eval_running and batch)
        )

        if should_judge:
            todo = batch[: args.batch_size]
            append_log(log_path, {"event": "judge_batch_start", "batch": len(todo), "pending": len(batch), "completed": completed})
            judged = []
            for idx, row in enumerate(todo, start=1):
                sample_id = str(row.get("sample_id"))
                out = dict(row)
                out["judge_sidecar_attempt"] = attempts_by_id.get(sample_id, 0) + 1
                try:
                    out["judge"] = judge_prediction_row(
                        out,
                        model=args.model,
                        base_url=args.base_url,
                        timeout=args.timeout,
                        max_retries=args.max_retries,
                        max_tokens=args.max_tokens,
                    )
                    out.pop("judge_error", None)
                except Exception as exc:
                    out["judge_error"] = str(exc)
                judged.append(out)
                append_log(log_path, {"event": "judged", "idx": idx, "batch": len(todo), "sample_id": sample_id, "ok": isinstance(out.get("judge"), dict), "error": bool(out.get("judge_error"))})
            append_jsonl(sidecar_path, judged)
            sidecar_rows.extend(judged)
            sidecar_by_id, attempts_by_id = sidecar_index(sidecar_rows)

        combined = combine_rows(rows, sidecar_by_id)
        write_snapshot(latest_dir, combined, args.baseline, args.expected_total)
        judged_count = sum(1 for row in combined if isinstance(row.get("judge"), dict))
        errors = sum(1 for row in combined if row.get("judge_error"))
        append_log(log_path, {"event": "poll", "completed": completed, "judged": judged_count, "judge_errors": errors, "pending": len(pending_rows(rows, sidecar_by_id, attempts_by_id, args.max_attempts)), "eval_running": eval_running})

        if completed >= args.expected_total and not eval_running:
            final_rows = combine_rows(load_latest_predictions(args.root), sidecar_by_id)
            write_snapshot(final_dir, final_rows, args.baseline, args.expected_total)
            append_log(log_path, {"event": "final", "completed": len(final_rows), "final_dir": str(final_dir)})
            break

        time.sleep(args.poll_seconds)


def load_latest_predictions(root: Path) -> list[dict[str, Any]]:
    by_id: dict[str, tuple[float, int, dict[str, Any]]] = {}
    order = 0
    for pred in sorted(root.glob("**/predictions.jsonl")):
        rel = pred.relative_to(root).parts
        if rel and (rel[0].startswith("merged") or rel[0] == "judge_sidecar"):
            continue
        mtime = pred.stat().st_mtime
        for line in pred.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(row.get("sample_id"))
            if sid == "None":
                sid = f"__missing_{order}"
            candidate = (mtime, order, row)
            if sid not in by_id or candidate[:2] >= by_id[sid][:2]:
                by_id[sid] = candidate
            order += 1
    rows = [item[2] for item in by_id.values()]
    rows.sort(key=sample_sort_key)
    return rows


def sample_sort_key(row: dict[str, Any]) -> int:
    value = str(row.get("sample_id"))
    return int(value) if value.isdigit() else 10**9


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sidecar_index(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    by_id: dict[str, dict[str, Any]] = {}
    attempts: dict[str, int] = {}
    for row in rows:
        sid = str(row.get("sample_id"))
        attempts[sid] = attempts.get(sid, 0) + 1
        by_id[sid] = row
    return by_id, attempts


def pending_rows(rows: list[dict[str, Any]], sidecar_by_id: dict[str, dict[str, Any]], attempts_by_id: dict[str, int], max_attempts: int) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for row in rows:
        sid = str(row.get("sample_id"))
        sidecar = sidecar_by_id.get(sid)
        if isinstance(row.get("judge"), dict) or (sidecar and isinstance(sidecar.get("judge"), dict)):
            continue
        if attempts_by_id.get(sid, 0) >= max_attempts:
            continue
        pending.append(row)
    return pending


def combine_rows(rows: list[dict[str, Any]], sidecar_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        sid = str(row.get("sample_id"))
        merged = dict(row)
        sidecar = sidecar_by_id.get(sid)
        if sidecar:
            if isinstance(sidecar.get("judge"), dict):
                merged["judge"] = sidecar["judge"]
                merged.pop("judge_error", None)
            elif sidecar.get("judge_error") and not isinstance(merged.get("judge"), dict):
                merged["judge_error"] = sidecar["judge_error"]
        out.append(merged)
    out.sort(key=sample_sort_key)
    return out


def write_snapshot(path: Path, rows: list[dict[str, Any]], baseline: str, expected_total: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_jsonl(path / "predictions.jsonl", rows)
    summary = summarize(rows, baseline=baseline)
    add_judge_summary(summary, rows)
    summary["unique_sample_ids"] = len({str(row.get("sample_id")) for row in rows})
    summary["expected_total"] = expected_total
    summary["missing_count"] = max(0, expected_total - summary["unique_sample_ids"])
    (path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_running() -> bool:
    import subprocess
    result = subprocess.run(["bash", "-lc", "pgrep -af 'scripts/evaluate.py' || true"], capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return any("watch_judge_predictions.py" not in line for line in lines)


def append_log(path: Path, item: dict[str, Any]) -> None:
    item = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **item}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
