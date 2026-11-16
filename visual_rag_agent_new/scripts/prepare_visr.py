#!/usr/bin/env python
"""VisR-Bench questions as UniDoc-RL ships them (HF DeepGlint-AI/UniDoc-RL rl/train.parquet rows with data_source
"visbench": one gold page per question, one evidence box in 0-1000 coordinates, a crop_required flag; pages in
rl/images.tar.gz named <doc>_<k>.<ext> with k 0-based) -> the layout the agent and the retriever index use:

  OUT/pages/<doc>/page_NN.<ext>   hard links to the extracted images (page NN = k + 1)
  OUT/train.jsonl                 one row per question
  OUT/smoke30.jsonl               30 seeded questions, a third of them crop_required

Row: sample_id, source_index, question, answer, deck_name, evidence_pages (1-based), hop ("single"), num_pages,
gold_bbox_1000 ([x1, y1, x2, y2] on the 0-1000 scale of the gold page), crop_required.

usage: python -I scripts/prepare_visr.py --parquet rl_train.parquet --images-dir RAW/<dir with the page files> --out-dir OUT
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

NAME = re.compile(r"^(.+)_(\d+)\.(jpg|jpeg|png|webp)$", re.IGNORECASE)


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
    zero_based = any(0 in ps for ps in pages.values())
    if not zero_based:
        raise ValueError("no page 0 anywhere: expected 0-based page files")
    for doc, ps in pages.items():
        d = out / "pages" / doc
        d.mkdir(parents=True, exist_ok=True)
        for k, src in ps.items():
            dst = d / f"page_{k + 1:02d}{src.suffix.lower()}"
            if not dst.exists():
                os.link(src, dst)

    rows, missing = [], Counter()
    table = pq.read_table(args.parquet).to_pylist()
    for i, r in enumerate(x for x in table if x["data_source"] == "visbench"):
        (page_name,) = r["reference_page"]
        doc, k = page_name.rsplit("_", 1)
        k = int(k)
        boxes = json.loads(r["bboxes"])[page_name]
        crop = bool(json.loads(r["crop_required"])[page_name])
        if doc not in pages or k not in pages[doc]:
            missing[doc] += 1
            continue
        if len(boxes) != 1 or not all(0 <= v <= 1000 for v in boxes[0]):
            raise ValueError(f"row {i}: unexpected boxes {boxes}")
        rows.append({"sample_id": f"visr_{i}", "source_index": r["index"],
                     "question": str(r["extra_info"]["question"]).strip(), "answer": str(r["reward_model"]["ground_truth"]).strip(),
                     "deck_name": doc, "evidence_pages": [k + 1], "hop": "single", "num_pages": len(pages[doc]),
                     "gold_bbox_1000": [int(v) for v in boxes[0]], "crop_required": crop})
    with (out / "train.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rng = random.Random(0)
    need = [r for r in rows if r["crop_required"]]
    rest = [r for r in rows if not r["crop_required"]]
    smoke = rng.sample(need, min(10, len(need))) + rng.sample(rest, 30 - min(10, len(need)))
    with (out / "smoke30.jsonl").open("w", encoding="utf-8") as f:
        for row in sorted(smoke, key=lambda r: int(r["sample_id"].split("_")[1])):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    used = {r["deck_name"] for r in rows}
    contiguous = sum(sorted(ps) == list(range(len(ps))) for ps in pages.values())
    stats = {"questions": len(rows), "questions without their page file": sum(missing.values()), "crop_required": len(need),
             "docs": len(pages), "docs with questions": len(used), "pages": sum(len(v) for v in pages.values()),
             "docs with contiguous pages from 0": contiguous, "pages per doc max": max(len(v) for v in pages.values()),
             "answer words p50": sorted(len(r["answer"].split()) for r in rows)[len(rows) // 2]}
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
