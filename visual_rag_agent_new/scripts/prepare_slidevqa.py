#!/usr/bin/env python
"""SlideVQA parquet (NTT-hil-insight/SlideVQA, one row per QA with the deck's 20 page images) ->
the layout the agent and the retriever index use:

  OUT/pages/<split>/<deck>/page_NN.<ext>   each deck page once, stored bytes as-is
  OUT/<split>.jsonl                        one row per QA
  OUT/stats.json                           counts per split + deck overlap between splits

Row: sample_id, qa_id, split, deck_name, deck_url, question, answer, arithmetic_expression,
evidence_pages (1-based, sorted), hop ("multi" when the gold spans >= 2 pages), num_pages.

usage: python -I scripts/prepare_slidevqa.py --raw-dir RAW --out-dir OUT --splits train val
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

MAX_PAGES = 20


def image_ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    raise ValueError(f"unknown image header {data[:12]!r}")


def write_page(deck_dir: Path, page: int, data: bytes) -> None:
    path = deck_dir / f"page_{page:02d}{image_ext(data)}"
    if path.exists() and path.stat().st_size == len(data):
        return
    deck_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def deck_dir_name(deck: str) -> str:
    name = deck.strip().replace("/", "_")
    if not name or name.startswith("."):
        raise ValueError(f"bad deck name {deck!r}")
    return name


def process_file(job: tuple[str, str, str]) -> list[dict]:
    import pyarrow.parquet as pq

    path, split, pages_root = job
    rows: list[dict] = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
        d = batch.to_pydict()
        for i in range(batch.num_rows):
            deck = deck_dir_name(str(d["deck_name"][i]))
            pages = []
            for k in range(1, MAX_PAGES + 1):
                img = d.get(f"page_{k}", [None] * batch.num_rows)[i]
                if img and img.get("bytes"):
                    write_page(Path(pages_root) / deck, k, img["bytes"])
                    pages.append(k)
            if pages != list(range(1, len(pages) + 1)):
                raise ValueError(f"{deck}: non-contiguous pages {pages}")
            evidence = sorted({int(p) for p in d["evidence_pages"][i] or []})
            if not evidence or max(evidence) > len(pages) or min(evidence) < 1:
                raise ValueError(f"qa {d['qa_id'][i]}: evidence {evidence} outside 1..{len(pages)}")
            rows.append({
                "sample_id": f"slidevqa_{split}_{d['qa_id'][i]}",
                "qa_id": int(d["qa_id"][i]),
                "split": split,
                "deck_name": deck,
                "deck_url": d.get("deck_url", [""] * batch.num_rows)[i],
                "question": str(d["question"][i]).strip(),
                "answer": str(d["answer"][i]).strip(),
                "arithmetic_expression": d.get("arithmetic_expression", [None] * batch.num_rows)[i],
                "evidence_pages": evidence,
                "hop": "multi" if len(evidence) >= 2 else "single",
                "num_pages": len(pages),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    raw, out = Path(args.raw_dir), Path(args.out_dir)
    stats: dict = {"splits": {}}
    decks_by_split: dict[str, set[str]] = {}
    for split in args.splits:
        files = sorted((raw / "data").glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"no {split} parquet under {raw / 'data'}")
        pages_root = out / "pages" / split
        with Pool(min(args.workers, len(files))) as pool:
            parts = pool.map(process_file, [(str(f), split, str(pages_root)) for f in files])
        rows = sorted((r for part in parts for r in part), key=lambda r: r["qa_id"])
        ids = Counter(r["sample_id"] for r in rows)
        dup = [k for k, v in ids.items() if v > 1]
        if dup:
            raise ValueError(f"{split}: duplicate qa ids {dup[:5]}")
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        decks = {r["deck_name"] for r in rows}
        decks_by_split[split] = decks
        n_files = sum(1 for _ in pages_root.rglob("page_*"))
        stats["splits"][split] = {
            "qa": len(rows),
            "decks": len(decks),
            "page_files": n_files,
            "hop": dict(Counter(r["hop"] for r in rows)),
            "num_evidence": dict(sorted(Counter(len(r["evidence_pages"]) for r in rows).items())),
            "pages_per_deck": dict(sorted(Counter(r["num_pages"] for r in rows).items())),
            "with_arithmetic": sum(1 for r in rows if r["arithmetic_expression"] not in (None, "", "None")),
        }
        print(split, json.dumps(stats["splits"][split]), flush=True)
    names = sorted(decks_by_split)
    stats["deck_overlap"] = {f"{a}&{b}": len(decks_by_split[a] & decks_by_split[b])
                             for i, a in enumerate(names) for b in names[i + 1:]}
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("deck_overlap", stats["deck_overlap"])


if __name__ == "__main__":
    main()
