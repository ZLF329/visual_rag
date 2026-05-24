#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an in-progress evaluate.py output directory.")
    parser.add_argument("--root", default="outputs/eval_notebook/main_agent", help="Directory containing timestamped eval runs.")
    parser.add_argument("--run", default=None, help="Specific timestamped run directory.")
    parser.add_argument("--expected", type=int, default=200)
    parser.add_argument("--tail", type=int, default=3)
    args = parser.parse_args()

    run = Path(args.run) if args.run else latest_run(Path(args.root))
    if run is None:
        print(json.dumps({"status": "no_run_found", "root": args.root}, ensure_ascii=False, indent=2))
        return

    sample_dirs = sorted(
        [p for p in run.iterdir() if p.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else 10**9,
    )
    done = [p for p in sample_dirs if (p / "answer.txt").exists() and (p / "trace.jsonl").exists()]
    predictions_path = run / "predictions.jsonl"
    prediction_rows = read_jsonl_lines(predictions_path) if predictions_path.exists() else []
    done_source = "predictions.jsonl" if predictions_path.exists() else "artifacts"
    done_count = len(prediction_rows) if predictions_path.exists() else len(done)
    payload = {
        "run": str(run),
        "expected": args.expected,
        "sample_dirs": len(sample_dirs),
        "artifact_done": len(done),
        "done": done_count,
        "done_source": done_source,
        "percent": round(100 * done_count / args.expected, 2) if args.expected else None,
        "predictions_written": predictions_path.exists(),
        "prediction_rows": len(prediction_rows),
        "summary_written": (run / "summary.json").exists(),
    }
    if not predictions_path.exists() and done:
        payload["warning"] = "predictions.jsonl is missing; artifact_done can include failed or partial samples"
    if prediction_rows:
        last_prediction = prediction_rows[-1]
        payload["last_prediction"] = {
            "sample_id": last_prediction.get("sample_id"),
            "terminated_by": last_prediction.get("terminated_by"),
            "has_judge": "judge" in last_prediction,
            "judge_error": last_prediction.get("judge_error"),
        }
    if done:
        last = done[-1]
        trace_lines = (last / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        terminated_by = None
        if trace_lines:
            try:
                last_step = json.loads(trace_lines[-1])
                terminated_by = last_step.get("result", {}).get("action") or last_step.get("step")
            except Exception:
                terminated_by = "unknown"
        payload["last_done"] = {
            "sample_id": last.name,
            "trace_steps": len(trace_lines),
            "terminated_by": terminated_by,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def read_jsonl_lines(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"decode_error": True, "raw": line[:200]})
    return rows


def latest_run(root: Path) -> Path | None:
    if not root.exists():
        return None
    runs = [p for p in root.iterdir() if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


if __name__ == "__main__":
    main()
