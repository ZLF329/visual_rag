#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment a SlideVQA retriever index with pages recoverable from HF parquet image bytes.")
    parser.add_argument("--base-index", default="/scratch/punim0614/lifuzhang/visual_rag_agent/data/indexes/slidevqa_train_balanced_2000")
    parser.add_argument("--grpo-dir", default="/scratch/punim0614/lifuzhang/visual_rag_agent/data/grpo/visor_slidevqa_balanced_800")
    parser.add_argument("--hf-parquet-dir", default="/scratch/punim0614/lifuzhang/hf_data/NTT-hil-insight-SlideVQA/data")
    parser.add_argument("--page-output", default="/scratch/punim0614/lifuzhang/visual_rag_agent/data/corpora/slidevqa_ra800_parquet_full_decks/pages")
    parser.add_argument("--output-index", default="/scratch/punim0614/lifuzhang/visual_rag_agent/data/indexes/slidevqa_train_balanced_2000_plus_ra800_full_decks")
    parser.add_argument("--model", default="/scratch/punim0614/lifuzhang/models/Qwen3-VL-Embedding-8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_index = Path(args.base_index)
    base_entries = json.loads((base_index / "filenames.json").read_text(encoding="utf-8"))
    base_labels = {str(item.get("page_label") if isinstance(item, dict) else Path(item).stem) for item in base_entries}
    base_embeddings = np.load(base_index / "embeddings.npy").astype("float32")

    target_refs = collect_target_refs(Path(args.grpo_dir))
    missing_refs = sorted(target_refs - base_labels)
    missing_decks = {label.split("/", 1)[0] for label in missing_refs if "/" in label}

    page_root = Path(args.page_output)
    extracted = extract_pages_from_hf_parquets(
        hf_dir=Path(args.hf_parquet_dir),
        decks=missing_decks,
        existing_labels=base_labels,
        page_root=page_root,
        dry_run=args.dry_run,
    )
    extracted_labels = {item["page_label"] for item in extracted}
    still_missing_refs = sorted(set(missing_refs) - extracted_labels)

    summary = {
        "base_index": str(base_index),
        "base_entries": len(base_entries),
        "target_reference_pages": len(target_refs),
        "missing_reference_pages_before": len(missing_refs),
        "missing_decks": len(missing_decks),
        "extracted_new_pages": len(extracted),
        "missing_reference_pages_after_extract": len(still_missing_refs),
        "page_output": str(page_root),
        "output_index": str(args.output_index),
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if extracted:
        from src.retriever import GVEEmbedder, normalize_rows

        embedder = GVEEmbedder(
            model_path=args.model,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        all_new_embeddings: list[np.ndarray] = []
        for start in range(0, len(extracted), args.batch_size):
            batch = extracted[start : start + args.batch_size]
            images = []
            for item in batch:
                with Image.open(item["image_path"]) as image:
                    images.append(image.convert("RGB").copy())
            all_new_embeddings.append(embedder.embed_images(images))
            print(json.dumps({"embedded": min(start + len(batch), len(extracted)), "total": len(extracted)}, ensure_ascii=False), flush=True)
        new_embeddings = normalize_rows(np.concatenate(all_new_embeddings, axis=0)).astype("float32")
        merged_embeddings = normalize_rows(np.concatenate([base_embeddings, new_embeddings], axis=0)).astype("float32")
    else:
        merged_embeddings = base_embeddings

    output_index = Path(args.output_index)
    output_index.mkdir(parents=True, exist_ok=True)
    merged_entries = list(base_entries) + extracted
    np.save(output_index / "embeddings.npy", merged_embeddings)
    (output_index / "filenames.json").write_text(json.dumps(merged_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    summary.update({"output_entries": len(merged_entries), "embedding_shape": list(merged_embeddings.shape)})
    (output_index / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def collect_target_refs(grpo_dir: Path) -> set[str]:
    import pandas as pd

    refs: set[str] = set()
    for split in ["train", "val"]:
        df = pd.read_parquet(grpo_dir / f"{split}.parquet", columns=["extra_info"])
        for extra in df["extra_info"]:
            labels = as_list((extra or {}).get("page_labels"))
            refs.update(str(label) for label in labels if str(label))
    return refs


def extract_pages_from_hf_parquets(*, hf_dir: Path, decks: set[str], existing_labels: set[str], page_root: Path, dry_run: bool) -> list[dict[str, str]]:
    import pandas as pd

    extracted_by_label: dict[str, dict[str, str]] = {}
    files = sorted(hf_dir.glob("train-*.parquet"))
    page_columns = [f"page_{i}" for i in range(1, 21)]
    for parquet_path in files:
        df = pd.read_parquet(parquet_path, columns=["deck_name", *page_columns])
        for _, row in df.iterrows():
            deck_name = str(row.get("deck_name") or "")
            if deck_name not in decks:
                continue
            for page in range(1, 21):
                image_obj = row.get(f"page_{page}")
                if image_obj is None:
                    continue
                page_label = f"{deck_name}/page_{page:02d}"
                if page_label in existing_labels or page_label in extracted_by_label:
                    continue
                out_path = page_root / deck_name / f"page_{page:02d}.png"
                extracted_by_label[page_label] = {"image_path": str(out_path), "page_label": page_label}
                if not dry_run:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    save_image(image_obj, out_path)
    return [extracted_by_label[key] for key in sorted(extracted_by_label)]


def save_image(image_obj: Any, out_path: Path) -> None:
    if isinstance(image_obj, dict) and image_obj.get("bytes") is not None:
        data = image_obj["bytes"]
    elif isinstance(image_obj, (bytes, bytearray)):
        data = image_obj
    else:
        raise ValueError(f"unsupported image object: {type(image_obj)}")
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGB").save(out_path)


def normalize_pages(value: Any) -> list[int]:
    pages: list[int] = []
    for item in as_list(value):
        try:
            page = int(item)
        except (TypeError, ValueError):
            continue
        if page > 0 and page not in pages:
            pages.append(page)
    return pages


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


if __name__ == "__main__":
    main()
