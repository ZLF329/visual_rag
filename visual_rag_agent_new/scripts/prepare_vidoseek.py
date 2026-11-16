#!/usr/bin/env python
"""ViDoSeek test set as UniDoc-RL ships it (HF DeepGlint-AI/UniDoc-RL test_data/: test_vidoseek.parquet +
images.tar.gz -> images/vidoseek/<sha1>_<page>.jpg, pages 1-based) -> the layout the agent and the retriever index use:

  OUT/pages/<doc>/page_NN.jpg   hard links to the extracted images
  OUT/test.jsonl                one row per question

Row: sample_id, source_index, question, answer, deck_name, evidence_pages (1-based), hop ("single" / "multi" from
ViDoSeek's data_source; every question has exactly one gold page), num_pages.

usage: python -I scripts/prepare_vidoseek.py --parquet test_vidoseek.parquet --images-dir RAW/images/vidoseek --out-dir OUT
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

NAME = re.compile(r"^([0-9a-f]{40})_(\d+)\.jpg$")
HOP = {"single_hop": "single", "multi_hop": "multi"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq

    out = Path(args.out_dir)
    pages: dict[str, dict[int, Path]] = {}
    for p in Path(args.images_dir).iterdir():
        m = NAME.match(p.name)
        if not m:
            raise ValueError(f"unexpected file {p.name!r}")
        pages.setdefault(m.group(1), {})[int(m.group(2))] = p
    for doc, ps in pages.items():
        if sorted(ps) != list(range(1, len(ps) + 1)):
            raise ValueError(f"{doc}: non-contiguous pages {sorted(ps)}")
        d = out / "pages" / doc
        d.mkdir(parents=True, exist_ok=True)
        for n, src in ps.items():
            dst = d / f"page_{n:02d}.jpg"
            if not dst.exists():
                os.link(src, dst)

    rows = []
    for i, r in enumerate(pq.read_table(args.parquet).to_pylist()):
        doc, ev = str(r["file_name"]), sorted({int(p) for p in r["reference_page"]})
        if doc not in pages or not ev or min(ev) < 1 or max(ev) > len(pages[doc]):
            raise ValueError(f"row {i}: doc {doc} evidence {ev} outside its {len(pages.get(doc, {}))} pages")
        rows.append({"sample_id": f"vidoseek_{i}", "source_index": r["index"],
                     "question": str(r["extra_info"]["question"]).strip(), "answer": str(r["reward_model"]["ground_truth"]).strip(),
                     "deck_name": doc, "evidence_pages": ev, "hop": HOP[r["data_source"]], "num_pages": len(pages[doc])})
    with (out / "test.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    used = {r["deck_name"] for r in rows}
    stats = {"questions": len(rows), "hop": dict(Counter(r["hop"] for r in rows)), "docs": len(pages),
             "docs with questions": len(used), "pages": sum(len(v) for v in pages.values()),
             "pages per doc max": max(len(v) for v in pages.values())}
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
