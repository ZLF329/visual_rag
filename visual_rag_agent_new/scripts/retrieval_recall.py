#!/usr/bin/env python
"""Gold-page recall of the raw question against a retriever index, flat corpus vs deck-scoped.
A quick check of how hard first-hop retrieval is before running the teacher.

usage: python scripts/retrieval_recall.py --dataset-file train.jsonl --index IDX --model COLQWEN \
           --num-samples 500 --k 1 5 10 --scopes flat deck
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from src.dataset import page_key, read_jsonl, reference_labels, retrieval_deck
from src.retriever import make_retriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-file", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--scopes", nargs="+", default=["flat", "deck"])
    ap.add_argument("--bank-device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.dataset_file))
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.num_samples]
    retriever = make_retriever(args.model, args.index, bank_device=args.bank_device)
    top = max(args.k)
    report: dict = {"num_samples": len(rows)}
    for scope in args.scopes:
        sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            ranked = [page_key(lbl) for lbl in retriever.search_labels(row["question"], top_k=top,
                                                                      deck_name=retrieval_deck(row, scope))]
            gold = {page_key(lbl) for lbl in reference_labels(row)}
            for group in ("all", row.get("hop", "?")):
                counts[group] += 1
                for k in args.k:
                    hit = gold & set(ranked[:k])
                    sums[group][f"recall@{k}"] += len(hit) / len(gold)
                    sums[group][f"all_gold@{k}"] += float(hit == gold)
                    sums[group][f"any_gold@{k}"] += float(bool(hit))
        report[scope] = {g: {m: round(v / counts[g], 4) for m, v in ms.items()} | {"n": counts[g]}
                         for g, ms in sums.items()}
        print(scope, json.dumps(report[scope], indent=1), flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
