#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def trace_counts(trace: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for step in trace or []:
        kind = str(step.get("step") or "")
        counts[kind] += 1
        if kind == "update_graph":
            decision = step.get("decision") or {}
            dtype = str(decision.get("type") or "").lower()
            if dtype:
                counts[f"update_{dtype}"] += 1
    return dict(counts)


def get_answer(record: dict[str, Any]) -> str:
    return str(record.get("answer") or record.get("final_answer") or "").strip()


def terminated_by(record: dict[str, Any]) -> str:
    return str(record.get("terminated_by") or record.get("stop_reason") or "").strip()


def judge_correct(record: dict[str, Any]) -> bool | None:
    judge = record.get("judge") if isinstance(record.get("judge"), dict) else record.get("validator")
    if isinstance(judge, dict):
        return judge.get("correct") is True or judge.get("score") == 1
    return None


def evidence_pages(record: dict[str, Any]) -> list[Any]:
    pages = (
        record.get("evidence_pages")
        or record.get("reference_pages")
        or record.get("reference_page_labels")
        or []
    )
    return pages if isinstance(pages, list) else []


def is_multi(record: dict[str, Any]) -> bool:
    hop = str(record.get("hop_type") or record.get("source_hop_type") or "").lower()
    if hop == "multi":
        return True
    return len(evidence_pages(record)) > 1


def root_sufficient(record: dict[str, Any]) -> bool:
    if terminated_by(record) in {"answer", "decide_answer"}:
        return True
    mem = record.get("memory") or {}
    state = mem.get("evidence_state") if isinstance(mem, dict) else None
    if isinstance(state, dict):
        missing = state.get("missing_requirements") or []
        return not missing
    return False


def score_record(record: dict[str, Any]) -> tuple[int, list[str], dict[str, int]]:
    flags: list[str] = []
    trace = record.get("trace") or []
    counts = trace_counts(trace)
    score = 100

    term = terminated_by(record)
    if term not in {"answer", "decide_answer"}:
        flags.append("not_answer_terminated")
        score -= 80
    if not get_answer(record):
        flags.append("empty_answer")
        score -= 80
    if "<answer from observation>" in get_answer(record).lower():
        flags.append("placeholder_answer")
        score -= 50

    jc = judge_correct(record)
    if jc is False:
        flags.append("judge_incorrect")
        score -= 80
    elif jc is None:
        flags.append("judge_missing")
        score -= 15

    if not root_sufficient(record):
        flags.append("root_not_sufficient")
        score -= 25

    search_count = counts.get("search", 0)
    reject_count = counts.get("update_reject", 0)
    expand_count = counts.get("update_expand", 0)
    accept_count = counts.get("update_accept", 0)
    crop_count = counts.get("crop", 0)
    error_count = sum(v for k, v in counts.items() if k.endswith("_error"))

    if error_count:
        flags.append("trace_error")
        score -= 60
    if expand_count < 1:
        flags.append("no_expand")
        score -= 18
    if accept_count < 2:
        flags.append("lt2_accepts")
        score -= 20
    if reject_count >= 2:
        flags.append("many_rejects")
        score -= 12 * reject_count
    if search_count >= 5:
        flags.append("long_search")
        score -= 10
    if len(trace) >= 12:
        flags.append("long_trajectory")
        score -= 10

    # Crop is allowed to be full quality. Only bad crop behavior is penalized.
    if crop_count >= 4:
        flags.append("many_crops")
        score -= 8 * (crop_count - 3)
    if counts.get("crop_error", 0):
        flags.append("crop_error")
        score -= 60

    queries = []
    for step in trace:
        if step.get("step") == "search":
            query = str(step.get("query") or "").strip().lower()
            if query:
                queries.append(query)
    if len(queries) != len(set(queries)):
        flags.append("repeated_search")
        score -= 15

    if not is_multi(record):
        flags.append("not_multi")
        score -= 100

    return max(0, min(100, score)), flags, {
        "search_count": search_count,
        "reject_count": reject_count,
        "expand_count": expand_count,
        "accept_count": accept_count,
        "crop_count": crop_count,
        "trace_len": len(trace),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select adjusted score=100 multi active-graph trajectories; crop itself is not penalized."
    )
    parser.add_argument(
        "--run-base",
        action="append",
        default=[],
        help="Directory to recursively scan for kept_trajectories.jsonl/trajectories.jsonl.",
    )
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paths: set[Path] = set()
    for base in args.run_base:
        base_path = Path(base)
        for name in ("kept_trajectories.jsonl", "trajectories.jsonl"):
            paths.update(base_path.rglob(name))
    for pattern in args.input_glob:
        paths.update(Path(path) for path in glob.glob(pattern, recursive=True))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "multi_audit_crop_allowed.jsonl"
    best_path = out_dir / "multi_best_by_sample_crop_allowed.jsonl"
    selected_path = out_dir / "multi_score100_crop_allowed.jsonl"

    all_rows = []
    best: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        for idx, record in enumerate(iter_jsonl(path)):
            if not is_multi(record):
                continue
            score, flags, counts = score_record(record)
            row = {
                "sample_id": str(record.get("sample_id") or record.get("row_index") or ""),
                "source_file": str(path),
                "source_index": idx,
                "score": score,
                "flags": flags,
                "crop_count": counts["crop_count"],
                "query": record.get("query") or record.get("question"),
                "gold_answer": record.get("gold_answer") or record.get("reference_answer"),
                "answer": get_answer(record),
                "terminated_by": terminated_by(record),
                "judge_correct": judge_correct(record),
                "evidence_pages": evidence_pages(record),
                **counts,
                "record": record,
            }
            all_rows.append(row)
            old = best.get(row["sample_id"])
            if old is None or (row["score"], row["crop_count"], -row["trace_len"]) > (
                old["score"],
                old["crop_count"],
                -old["trace_len"],
            ):
                best[row["sample_id"]] = row

    best_rows = sorted(
        best.values(),
        key=lambda row: (-row["score"], -row["crop_count"], row["trace_len"], row["sample_id"]),
    )
    selected = [row for row in best_rows if row["score"] == 100]

    for path, rows in ((audit_path, all_rows), (best_path, best_rows), (selected_path, selected)):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    flag_counts = Counter(flag for row in best_rows for flag in row["flags"])
    with_crop = sum(1 for row in selected if row["crop_count"] > 0)
    summary = {
        "scanned_files": [str(path) for path in sorted(paths)],
        "all_multi_records": len(all_rows),
        "unique_multi_samples": len(best_rows),
        "score100_selected": len(selected),
        "score100_with_crop": with_crop,
        "score100_crop_ratio": with_crop / len(selected) if selected else 0,
        "score100_crop_actions": sum(row["crop_count"] for row in selected),
        "best_flag_counts": dict(flag_counts),
        "audit_path": str(audit_path),
        "best_path": str(best_path),
        "selected_path": str(selected_path),
        "validation_mechanism": (
            "DeepSeek judge fields when present plus structural active-graph trace rubric; "
            "crop itself is not penalized."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
