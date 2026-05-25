#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_FILE = (
    "outputs/sft_trajectories/gpt5mini_20260525_004442/kept_sft_trajectories.jsonl"
)
DEFAULT_IMAGE_ROOT = "data/corpora/slidevqa_sft/pages"
DEFAULT_OUTPUT_DIR = "data/grpo/visor_slidevqa"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple multimodal verl GRPO dataset.")
    parser.add_argument("--sft-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--image-root", action="append", default=[DEFAULT_IMAGE_ROOT])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    resolver = ImageResolver([Path(p) for p in args.image_root])
    rows = build_rows(
        sft_file=Path(args.sft_file),
        resolver=resolver,
        max_images=args.max_images,
        limit=args.limit,
    )
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_size = min(args.val_size, max(1, len(rows) // 10))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(train_rows, output_dir / "train.parquet")
    write_parquet(val_rows, output_dir / "val.parquet")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "max_images": args.max_images,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_rows(
    *,
    sft_file: Path,
    resolver: "ImageResolver",
    max_images: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with sft_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            question = str(item.get("question") or item.get("query") or "").strip()
            answer = str(item.get("reference_answer") or "").strip()
            if not question or not answer:
                continue
            page_labels = pick_page_labels(item, max_images=max_images)
            images = []
            resolved_labels = []
            for label in page_labels:
                parsed = parse_page_label(label)
                if parsed is None:
                    continue
                path = resolver.resolve(parsed[0], parsed[1])
                if not path:
                    continue
                images.append({"bytes": Path(path).read_bytes(), "path": path})
                resolved_labels.append(label)
            if not images:
                continue
            prompt = (
                "Answer the visual document question using only the attached page image(s). "
                "Return only the concise final answer, without citations or extra explanation.\n\n"
                f"Question: {question}"
            )
            rows.append(
                {
                    "data_source": "slidevqa_visor",
                    "prompt": [{"role": "user", "content": prompt}],
                    "images": images,
                    "ability": "visual_document_qa",
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {
                        "split": "train",
                        "index": idx,
                        "sample_id": item.get("sample_id"),
                        "question": question,
                        "answer": answer,
                        "page_labels": resolved_labels,
                    },
                }
            )
    return rows


def pick_page_labels(item: dict[str, Any], *, max_images: int) -> list[str]:
    labels: list[str] = []
    for key in ["found_reference_page_labels", "reference_page_labels", "retrieved_page_labels"]:
        for label in item.get(key, []) or []:
            if label not in labels:
                labels.append(label)
            if len(labels) >= max_images:
                return labels
    return labels


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


class ImageResolver:
    def __init__(self, image_roots: list[Path]) -> None:
        self.image_roots = image_roots
        self.manifests: list[dict[str, str]] = []
        for root in image_roots:
            manifest = root / "manifest.json"
            if manifest.exists():
                self.manifests.append(json.loads(manifest.read_text(encoding="utf-8")))

    def resolve(self, deck: str, page_num: int) -> str | None:
        label = f"{deck}/page_{page_num:02d}"
        for manifest in self.manifests:
            candidate = manifest.get(label)
            if candidate and Path(candidate).exists():
                return str(Path(candidate))
        for root in self.image_roots:
            for candidate in [
                root / safe_name(deck) / f"page_{page_num:02d}.jpg",
                root / safe_name(deck) / f"page_{page_num:02d}.png",
                root / deck / f"page_{page_num:02d}.jpg",
                root / deck / f"page_{page_num:02d}.png",
            ]:
                if candidate.exists():
                    return str(candidate)
        return None


def parse_page_label(label: str) -> tuple[str, int] | None:
    match = re.match(r"(.+)/page_0*(\d+)$", str(label))
    if not match:
        return None
    return match.group(1), int(match.group(2))


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)[:160]


if __name__ == "__main__":
    main()
