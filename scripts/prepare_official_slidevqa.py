#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the official gated NTT-hil-insight/SlideVQA split from HuggingFace."
    )
    parser.add_argument("--output", default="data/corpora/slidevqa")
    parser.add_argument("--repo-id", default="NTT-hil-insight/SlideVQA")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--hf-cache-dir", default="/root/autodl-tmp/hf_data/NTT-hil-insight-SlideVQA")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--token", default=None, help="HF token with accepted gated dataset access.")
    parser.add_argument("--clean", action="store_true", help="Clear output/pages and output/test.jsonl first.")
    args = parser.parse_args()

    output = Path(args.output)
    if args.clean:
        clean_output(output)
    output.mkdir(parents=True, exist_ok=True)

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    files = list_split_files(
        repo_id=args.repo_id,
        split=args.split,
        endpoint=args.hf_endpoint,
        token=token,
    )
    if not files:
        raise RuntimeError(f"No parquet files found for split={args.split} in {args.repo_id}")

    rows_written = 0
    downloaded_files: list[str] = []
    with (output / f"{args.split}.jsonl").open("w", encoding="utf-8") as out_f:
        for filename in files:
            parquet_path = download_official_file(
                repo_id=args.repo_id,
                filename=filename,
                local_dir=Path(args.hf_cache_dir),
                endpoint=args.hf_endpoint,
                token=token,
            )
            downloaded_files.append(str(parquet_path))
            rows_written += export_rows_from_parquet(
                parquet_path=parquet_path,
                out_f=out_f,
                image_root=output / "pages",
                split=args.split,
                start_index=rows_written,
                max_rows=args.num_samples - rows_written,
            )
            if rows_written >= args.num_samples:
                break

    print(
        json.dumps(
            {
                "dataset_file": str(output / f"{args.split}.jsonl"),
                "image_dir": str(output / "pages"),
                "rows_written": rows_written,
                "downloaded_files": downloaded_files,
                "repo_id": args.repo_id,
                "split": args.split,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def clean_output(output: Path) -> None:
    for child in [output / "pages", output / "test.jsonl", output / "val.jsonl", output / "train.jsonl"]:
        if child.is_dir():
            shutil.rmtree(child)
        elif child.exists():
            child.unlink()


def list_split_files(*, repo_id: str, split: str, endpoint: str, token: str | None) -> list[str]:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    from huggingface_hub import HfApi

    api = HfApi(endpoint=endpoint)
    files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    prefix = f"data/{split}-"
    return sorted(path for path in files if path.startswith(prefix) and path.endswith(".parquet"))


def download_official_file(
    *,
    repo_id: str,
    filename: str,
    local_dir: Path,
    endpoint: str,
    token: str | None,
) -> Path:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(local_dir),
                token=token,
            )
        )
    except Exception as exc:
        message = str(exc)
        if "gated" in message.lower() or "403" in message:
            raise RuntimeError(
                "Official SlideVQA is gated. Accept access at "
                "https://huggingface.co/datasets/NTT-hil-insight/SlideVQA "
                "and run with HF_TOKEN set to an authorized token."
            ) from exc
        raise


def export_rows_from_parquet(
    *,
    parquet_path: Path,
    out_f: Any,
    image_root: Path,
    split: str,
    start_index: int,
    max_rows: int,
) -> int:
    import pyarrow.parquet as pq

    if max_rows <= 0:
        return 0

    table = pq.read_table(parquet_path)
    written = 0
    for row in table.to_pylist():
        if written >= max_rows:
            break
        sample_index = start_index + written
        deck_name = str(row.get("deck_name") or f"sample_{sample_index:06d}")
        page_paths = export_page_images(row=row, image_root=image_root, deck_name=deck_name)
        out = {
            "id": row.get("qa_id", sample_index),
            "qa_id": row.get("qa_id", sample_index),
            "deck_name": deck_name,
            "deck_url": row.get("deck_url", ""),
            "question": row.get("question", ""),
            "query": row.get("question", ""),
            "answer": row.get("answer", ""),
            "arithmetic_expression": row.get("arithmetic_expression", ""),
            "evidence_pages": row.get("evidence_pages", []),
            "answer_type": row.get("answer_type", ""),
            "reasoning_type": row.get("resoning_type", row.get("reasoning_type", "")),
            "page_images": page_paths,
            "split": split,
        }
        out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
        written += 1
    return written


def export_page_images(*, row: dict[str, Any], image_root: Path, deck_name: str) -> list[str]:
    page_paths: list[str] = []
    for page_idx in range(1, 21):
        key = f"page_{page_idx}"
        image_obj = row.get(key)
        if not image_obj:
            continue
        image_bytes = extract_image_bytes(image_obj)
        if not image_bytes:
            continue
        rel_path = Path(safe_name(deck_name)) / f"page_{page_idx:02d}.png"
        out_path = image_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            out_path.write_bytes(image_bytes)
        page_paths.append(str(rel_path))
    return page_paths


def extract_image_bytes(value: Any) -> bytes | None:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, bytes):
            return raw
    if isinstance(value, bytes):
        return value
    return None


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)[:160]


if __name__ == "__main__":
    main()
