#!/usr/bin/env python
"""Merge evaluation shards (run_eval.sh with START/N, one per GPU) into one run dir: predictions.jsonl (one row per
sample_id, in dataset order when the dataset is given) + summary.json computed exactly as evaluate.py does.
usage: python scripts/merge_eval_shards.py OUT_DIR SHARD_DIR [SHARD_DIR ...] [--dataset-file test.jsonl]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate import summarize
from src.dataset import extract_sample_id, read_jsonl
from src.judge import add_judge_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("shards", nargs="+")
    parser.add_argument("--dataset-file", default=None, help="checks that every sample is present, keeps its order")
    args = parser.parse_args()
    rows: dict[str, dict] = {}
    for shard in args.shards:
        for row in read_jsonl(Path(shard) / "predictions.jsonl"):
            assert row["sample_id"] not in rows, f"duplicate {row['sample_id']}"
            rows[row["sample_id"]] = row
    order = list(rows)
    if args.dataset_file:
        order = [extract_sample_id(r, i) for i, r in enumerate(read_jsonl(Path(args.dataset_file)))]
        missing = [s for s in order if s not in rows]
        assert not missing, f"{len(missing)} samples missing, e.g. {missing[:5]}"
    merged = [rows[s] for s in order]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    summary = summarize(merged)
    summary["retrieval_scope"] = merged[0].get("retrieval_scope", "flat") if merged else None
    summary["shards"] = list(args.shards)
    add_judge_summary(summary, merged)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
