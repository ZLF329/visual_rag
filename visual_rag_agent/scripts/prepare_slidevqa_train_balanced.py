#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


META_COLUMNS = [
    "deck_name",
    "deck_url",
    "qa_id",
    "question",
    "answer",
    "arithmetic_expression",
    "evidence_pages",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a balanced SlideVQA train subset from locally cached official parquet shards."
    )
    parser.add_argument("--hf-data-dir", default="/scratch/punim0614/lifuzhang/hf_data/NTT-hil-insight-SlideVQA/data")
    parser.add_argument("--output", default="data/corpora/slidevqa_train_balanced_2000")
    parser.add_argument("--single-target", type=int, default=1000)
    parser.add_argument("--multi-target", type=int, default=1000)
    parser.add_argument("--single-offset", type=int, default=0)
    parser.add_argument("--multi-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=["packed", "random"],
        default="packed",
        help="packed minimizes unique decks/pages; random maximizes sampling diversity.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    hf_data_dir = Path(args.hf_data_dir)
    output = Path(args.output)
    parquet_files = sorted(hf_data_dir.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no train parquet shards found under {hf_data_dir}")

    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates(parquet_files)
    selected_single, selected_multi = select_balanced(
        candidates,
        single_target=args.single_offset + args.single_target,
        multi_target=args.multi_offset + args.multi_target,
        seed=args.seed,
        strategy=args.strategy,
    )
    selected_single = selected_single[args.single_offset :]
    selected_multi = selected_multi[args.multi_offset :]
    selected = interleave(selected_single, selected_multi)

    selected_by_file: dict[Path, dict[int, dict[str, Any]]] = defaultdict(dict)
    for item in selected:
        selected_by_file[item["parquet_path"]][item["row_index"]] = item

    rows_by_hop = {"single": [], "multi": []}
    for parquet_path in sorted(selected_by_file):
        export_selected_rows(
            parquet_path=parquet_path,
            selected_rows=selected_by_file[parquet_path],
            output=output,
            rows_by_hop=rows_by_hop,
            batch_size=args.batch_size,
        )

    single_rows = sort_rows_like_selection(rows_by_hop["single"], selected_single)
    multi_rows = sort_rows_like_selection(rows_by_hop["multi"], selected_multi)
    combined_rows = interleave(single_rows, multi_rows)

    write_jsonl(output / "train_single.jsonl", single_rows)
    write_jsonl(output / "train_multi.jsonl", multi_rows)
    write_jsonl(output / "train.jsonl", combined_rows)

    summary = {
        "output": str(output),
        "split": "train",
        "strategy": args.strategy,
        "seed": args.seed,
        "single_offset": args.single_offset,
        "multi_offset": args.multi_offset,
        "single_target": args.single_target,
        "multi_target": args.multi_target,
        "single_rows": len(single_rows),
        "multi_rows": len(multi_rows),
        "total_rows": len(combined_rows),
        "unique_decks": len({row["deck_name"] for row in combined_rows}),
        "page_files": count_files(output / "pages", ".png"),
        "single_file": str(output / "train_single.jsonl"),
        "multi_file": str(output / "train_multi.jsonl"),
        "combined_file": str(output / "train.jsonl"),
        "page_root": str(output / "pages"),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def collect_candidates(parquet_files: list[Path]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    out: list[dict[str, Any]] = []
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=META_COLUMNS)
        for row_index, row in enumerate(table.to_pylist()):
            evidence_pages = normalize_pages(row.get("evidence_pages"))
            hop_type = "single" if len(evidence_pages) <= 1 else "multi"
            out.append(
                {
                    "parquet_path": parquet_path,
                    "row_index": row_index,
                    "deck_name": str(row.get("deck_name") or f"sample_{len(out):06d}"),
                    "qa_id": row.get("qa_id", len(out)),
                    "evidence_pages": evidence_pages,
                    "hop_type": hop_type,
                }
            )
    return out


def select_balanced(
    candidates: list[dict[str, Any]],
    *,
    single_target: int,
    multi_target: int,
    seed: int,
    strategy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_hop = {
        "single": [item for item in candidates if item["hop_type"] == "single"],
        "multi": [item for item in candidates if item["hop_type"] == "multi"],
    }
    if len(by_hop["single"]) < single_target:
        raise RuntimeError(f"only {len(by_hop['single'])} single-hop candidates available")
    if len(by_hop["multi"]) < multi_target:
        raise RuntimeError(f"only {len(by_hop['multi'])} multi-hop candidates available")

    if strategy == "random":
        rng = random.Random(seed)
        single = by_hop["single"][:]
        multi = by_hop["multi"][:]
        rng.shuffle(single)
        rng.shuffle(multi)
        return single[:single_target], multi[:multi_target]

    single, multi = select_packed(by_hop["single"], by_hop["multi"], single_target, multi_target)
    return single, multi


def select_packed(
    single_candidates: list[dict[str, Any]],
    multi_candidates: list[dict[str, Any]],
    single_target: int,
    multi_target: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_deck: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"single": [], "multi": []})
    for item in single_candidates:
        by_deck[item["deck_name"]]["single"].append(item)
    for item in multi_candidates:
        by_deck[item["deck_name"]]["multi"].append(item)

    decks = sorted(
        by_deck,
        key=lambda deck: (-len(by_deck[deck]["multi"]), -len(by_deck[deck]["single"]), deck),
    )
    selected_multi: list[dict[str, Any]] = []
    for deck in decks:
        need = multi_target - len(selected_multi)
        if need <= 0:
            break
        selected_multi.extend(by_deck[deck]["multi"][:need])

    selected_single: list[dict[str, Any]] = []
    selected_decks = {item["deck_name"] for item in selected_multi}
    deck_order = sorted(
        by_deck,
        key=lambda deck: (deck not in selected_decks, -len(by_deck[deck]["single"]), deck),
    )
    for deck in deck_order:
        need = single_target - len(selected_single)
        if need <= 0:
            break
        selected_single.extend(by_deck[deck]["single"][:need])

    return selected_single, selected_multi


def export_selected_rows(
    *,
    parquet_path: Path,
    selected_rows: dict[int, dict[str, Any]],
    output: Path,
    rows_by_hop: dict[str, list[dict[str, Any]]],
    batch_size: int,
) -> None:
    import pyarrow.parquet as pq

    if not selected_rows:
        return
    parquet = pq.ParquetFile(parquet_path)
    local_index = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            meta = selected_rows.get(local_index)
            if meta is not None:
                exported = export_row(row=row, meta=meta, output=output)
                rows_by_hop[meta["hop_type"]].append(exported)
            local_index += 1


def export_row(*, row: dict[str, Any], meta: dict[str, Any], output: Path) -> dict[str, Any]:
    deck_name = safe_name(str(row.get("deck_name") or meta["deck_name"]))
    page_paths = export_page_images(row=row, image_root=output / "pages", deck_name=deck_name)
    evidence_pages = normalize_pages(row.get("evidence_pages"))
    return {
        "id": row.get("qa_id", meta["qa_id"]),
        "qa_id": row.get("qa_id", meta["qa_id"]),
        "deck_name": deck_name,
        "deck_url": row.get("deck_url", ""),
        "question": row.get("question", ""),
        "query": row.get("question", ""),
        "answer": row.get("answer", ""),
        "arithmetic_expression": row.get("arithmetic_expression", ""),
        "evidence_pages": evidence_pages,
        "reference_pages": evidence_pages,
        "hop_type": meta["hop_type"],
        "evidence_count": len(evidence_pages),
        "page_images": page_paths,
        "split": "train",
        "source_parquet": meta["parquet_path"].name,
        "source_row_index": meta["row_index"],
    }


def export_page_images(*, row: dict[str, Any], image_root: Path, deck_name: str) -> list[str]:
    page_paths: list[str] = []
    for page_idx in range(1, 21):
        image_obj = row.get(f"page_{page_idx}")
        image_bytes = extract_image_bytes(image_obj)
        if not image_bytes:
            continue
        rel_path = Path(deck_name) / f"page_{page_idx:02d}.png"
        out_path = image_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():
            out_path.write_bytes(image_bytes)
        page_paths.append(str(rel_path))
    return page_paths


def normalize_pages(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(x) for x in value if str(x).strip()]
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


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


def interleave(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx in range(max(len(left), len(right))):
        if idx < len(left):
            out.append(left[idx])
        if idx < len(right):
            out.append(right[idx])
    return out


def sort_rows_like_selection(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {
        (item["parquet_path"].name, item["row_index"]): idx
        for idx, item in enumerate(selected)
    }
    return sorted(rows, key=lambda row: order[(row["source_parquet"], row["source_row_index"])])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_files(root: Path, suffix: str) -> int:
    return sum(1 for path in root.rglob(f"*{suffix}") if path.is_file()) if root.exists() else 0


if __name__ == "__main__":
    main()
