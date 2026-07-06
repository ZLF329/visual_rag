#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_MIMO25_RUN_ROOT = (
    "outputs/sft_trajectories/mimo25_train_balanced_1200_calllevel_20260527_172453"
)
DEFAULT_TRAIN_FILE = f"{DEFAULT_MIMO25_RUN_ROOT}/kept_balanced_1200.jsonl"
DEFAULT_PARQUET_ROOT = "/scratch/punim0614/lifuzhang/hf_data/NTT-hil-insight-SlideVQA/data"
DEFAULT_OUTPUT_ROOT = "data/corpora/slidevqa_sft/pages"
MIN_PAGE_NUM = 1
MAX_PAGE_NUM = 20


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export only the SlideVQA page images referenced by kept SFT trajectories."
    )
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--parquet-root", default=DEFAULT_PARQUET_ROOT)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    train_file = Path(args.train_file)
    parquet_root = Path(args.parquet_root)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest) if args.manifest else output_root / "manifest.json"

    needed = collect_needed_pages(train_file, limit=args.limit)
    written, missing_decks = export_needed_pages(
        needed=needed,
        parquet_root=parquet_root,
        split=args.split,
        output_root=output_root,
        manifest_path=manifest_path,
    )

    print(
        json.dumps(
            {
                "train_file": str(train_file),
                "parquet_root": str(parquet_root),
                "output_root": str(output_root),
                "manifest": str(manifest_path),
                "needed_decks": len(needed),
                "needed_pages": sum(len(v) for v in needed.values()),
                "written_or_existing_pages": written,
                "missing_decks": sorted(missing_decks)[:50],
                "missing_decks_count": len(missing_decks),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def collect_needed_pages(path: Path, *, limit: int | None = None) -> dict[str, set[int]]:
    needed: dict[str, set[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            for label in row.get("reference_page_labels", []):
                add_page_label(needed, label)
            for label in row.get("retrieved_page_labels", []):
                add_page_label(needed, label)
            for label in row.get("found_reference_page_labels", []):
                add_page_label(needed, label)
            for item in row.get("trace", []):
                for page in item.get("pages", []) or []:
                    if isinstance(page, dict):
                        label = page.get("page_label")
                        page_num = page.get("page_num")
                        if label and page_num:
                            deck = str(label).split("/page_")[0]
                            add_page(needed, deck, int(page_num))
                        elif label:
                            add_page_label(needed, label)
                    elif isinstance(page, str):
                        add_page_label(needed, page)
                if item.get("page"):
                    add_page_label(needed, item["page"])
    return needed


def add_page_label(needed: dict[str, set[int]], label: str) -> None:
    parsed = parse_page_label(label)
    if parsed is None:
        return
    deck, page_num = parsed
    add_page(needed, deck, page_num)


def add_page(needed: dict[str, set[int]], deck: str, page_num: int) -> None:
    if not deck or not (MIN_PAGE_NUM <= page_num <= MAX_PAGE_NUM):
        return
    needed.setdefault(deck, set()).add(page_num)


def parse_page_label(label: str) -> tuple[str, int] | None:
    match = re.match(r"(.+)/page_0*(\d+)(?:\.[A-Za-z0-9]+)?$", str(label))
    if not match:
        return None
    return match.group(1), int(match.group(2))


def export_needed_pages(
    *,
    needed: dict[str, set[int]],
    parquet_root: Path,
    split: str,
    output_root: Path,
    manifest_path: Path,
) -> tuple[int, set[str]]:
    import pyarrow.parquet as pq

    output_root.mkdir(parents=True, exist_ok=True)
    remaining = {deck: set(pages) for deck, pages in needed.items()}
    manifest: dict[str, str] = {}
    written = 0
    page_columns = sorted(
        {f"page_{page}" for pages in needed.values() for page in pages},
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )

    for parquet_path in sorted(parquet_root.glob(f"{split}-*.parquet")):
        if not remaining:
            break
        schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        columns = ["deck_name", *[column for column in page_columns if column in schema_names]]
        table = pq.read_table(parquet_path, columns=columns)
        for row in table.to_pylist():
            deck_name = str(row.get("deck_name") or "")
            if deck_name not in remaining:
                continue
            for page_num in sorted(remaining[deck_name]):
                image_obj = row.get(f"page_{page_num}")
                image_bytes = extract_image_bytes(image_obj)
                if not image_bytes:
                    continue
                rel_path = Path(safe_name(deck_name)) / f"page_{page_num:02d}.jpg"
                out_path = output_root / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if not out_path.exists():
                    out_path.write_bytes(image_bytes)
                manifest[f"{deck_name}/page_{page_num:02d}"] = str(out_path)
                written += 1
            remaining.pop(deck_name, None)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return written, set(remaining)


def extract_image_bytes(value: Any) -> bytes | None:
    if isinstance(value, dict):
        raw = value.get("bytes")
        return raw if isinstance(raw, bytes) else None
    return value if isinstance(value, bytes) else None


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)[:160]


if __name__ == "__main__":
    main()
