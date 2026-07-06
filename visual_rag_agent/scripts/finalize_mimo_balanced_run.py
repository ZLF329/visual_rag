#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge MiMo SFT trajectory shards with duplicate filtering.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--label", default="1200")
    parser.add_argument("--target", type=int, default=1200)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source descriptor dirname:hop_type:mode, where mode is kept or raw_kept.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    seen_deck_questions: set[tuple[str, str]] = set()
    source_counts: Counter[str] = Counter()
    hop_counts: Counter[str] = Counter()

    for source in args.source:
        dirname, hop_type, mode = parse_source(source)
        path = run_root / dirname / ("raw_trajectories.jsonl" if mode == "raw_kept" else "kept_sft_trajectories.jsonl")
        for row in load_jsonl(path):
            if mode == "raw_kept" and not row.get("keep"):
                continue
            sample_id = str(row.get("sample_id") or "")
            deck_question = (str(row.get("deck_name") or ""), str(row.get("question") or "").strip())
            reasons = []
            if sample_id and sample_id in seen_sample_ids:
                reasons.append("duplicate_sample_id")
            if deck_question[0] and deck_question[1] and deck_question in seen_deck_questions:
                reasons.append("duplicate_deck_question")
            if reasons:
                skipped.append(
                    {
                        "source_bucket": dirname,
                        "sample_id": sample_id,
                        "deck_name": deck_question[0],
                        "question": deck_question[1],
                        "reasons": reasons,
                    }
                )
                continue
            row["source_hop_type"] = hop_type
            row["source_bucket"] = dirname
            rows.append(row)
            source_counts[dirname] += 1
            hop_counts[hop_type] += 1
            if sample_id:
                seen_sample_ids.add(sample_id)
            if deck_question[0] and deck_question[1]:
                seen_deck_questions.add(deck_question)
            if len(rows) >= args.target:
                break
        if len(rows) >= args.target:
            break

    calls = []
    call_source_counts: Counter[str] = Counter()
    call_hop_counts: Counter[str] = Counter()
    for row in rows:
        bucket = str(row.get("source_bucket") or "")
        hop_type = str(row.get("source_hop_type") or "")
        for call in row.get("sft_calls") or []:
            call = dict(call)
            call["source_hop_type"] = hop_type
            call["source_bucket"] = bucket
            calls.append(call)
            call_source_counts[bucket] += 1
            call_hop_counts[hop_type] += 1

    out = run_root / f"kept_balanced_{args.label}.jsonl"
    calls_out = run_root / f"kept_balanced_{args.label}_sft_calls.jsonl"
    write_jsonl(out, rows)
    write_jsonl(calls_out, calls)

    key_counts = Counter((str(row.get("deck_name") or ""), str(row.get("question") or "").strip()) for row in rows)
    sample_counts = Counter(str(row.get("sample_id")) for row in rows if row.get("sample_id") is not None)
    summary = {
        "run_root": str(run_root),
        "merged_kept_file": str(out),
        "merged_kept_sft_calls_file": str(calls_out),
        "total_target": args.target,
        "total_kept": len(rows),
        "hop_counts": dict(sorted(hop_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "total_kept_sft_calls": len(calls),
        "hop_sft_call_counts": dict(sorted(call_hop_counts.items())),
        "source_sft_call_counts": dict(sorted(call_source_counts.items())),
        "duplicate_deck_question_pairs": sum(count - 1 for count in key_counts.values() if count > 1),
        "duplicate_sample_ids": sum(count - 1 for count in sample_counts.values() if count > 1),
        "skipped_duplicate_rows": len(skipped),
        "skipped_duplicate_reasons": dict(Counter(reason for row in skipped for reason in row["reasons"])),
        "sources": args.source,
    }
    (run_root / "balanced_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(run_root / "skipped_duplicate_rows.jsonl", skipped)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def parse_source(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"source must be dirname:hop_type:mode, got {value!r}")
    dirname, hop_type, mode = parts
    if mode not in {"kept", "raw_kept"}:
        raise ValueError(f"unsupported source mode {mode!r}")
    return dirname, hop_type, mode


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
