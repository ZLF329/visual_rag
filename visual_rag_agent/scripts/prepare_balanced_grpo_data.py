#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SFT_EXCLUDE_FILE = (
    "outputs/sft_trajectories/gpt5mini_20260525_153719/kept_sft_trajectories.page_local_clean.jsonl"
)
DEFAULT_PARQUET_ROOT = "/scratch/punim0614/lifuzhang/hf_data/NTT-hil-insight-SlideVQA/data"
DEFAULT_OUTPUT_DIR = "data/grpo/visor_slidevqa_balanced_800"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a balanced SlideVQA GRPO dataset from SlideVQA parquet, excluding SFT questions."
    )
    parser.add_argument("--sft-exclude-file", default=DEFAULT_SFT_EXCLUDE_FILE, help="Clean SFT trajectory JSONL used only to exclude overlapping questions; not used as RL training samples.")
    parser.add_argument("--extra-exclude-file", action="append", default=[], help="Additional JSONL SFT/QwenVL files to exclude by sample_id/qa_id and parsed question.")
    parser.add_argument("--parquet-root", default=DEFAULT_PARQUET_ROOT)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-single", type=int, default=400)
    parser.add_argument("--train-multi", type=int, default=400)
    parser.add_argument("--val-single", type=int, default=64)
    parser.add_argument("--val-multi", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=3)
    args = parser.parse_args()

    parquet_root = Path(args.parquet_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude = load_exclusions([Path(args.sft_exclude_file), *[Path(p) for p in args.extra_exclude_file]])
    candidates = collect_candidates(
        parquet_root=parquet_root,
        split=args.split,
        exclude=exclude,
        max_images=args.max_images,
    )
    selected = select_balanced(
        candidates=candidates,
        train_single=args.train_single,
        train_multi=args.train_multi,
        val_single=args.val_single,
        val_multi=args.val_multi,
        seed=args.seed,
    )

    train_rows = materialize_rows(
        selected["train"],
        max_images=args.max_images,
        data_source="slidevqa_grpo_train_balanced",
    )
    val_rows = materialize_rows(
        selected["val"],
        max_images=args.max_images,
        data_source="slidevqa_grpo_val_balanced",
    )

    write_parquet(train_rows, output_dir / "train.parquet")
    write_parquet(val_rows, output_dir / "val.parquet")

    summary = {
        "output_dir": str(output_dir),
        "source_split": args.split,
        "sft_exclude_file": args.sft_exclude_file,
        "exclude_key_count": len(exclude["deck_question_keys"]),
        "exclude_question_count": len(exclude["questions"]),
        "exclude_id_count": len(exclude["ids"]),
        "available_candidates": count_by_hop(candidates),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_hop_counts": count_by_hop(train_rows),
        "val_hop_counts": count_by_hop(val_rows),
        "max_images": args.max_images,
        "seed": args.seed,
        "overlap_with_sft_exclude_file": count_overlap(train_rows + val_rows, exclude),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_exclusions(paths: list[Path]) -> dict[str, set[Any]]:
    exclusions: dict[str, set[Any]] = {
        "deck_question_keys": set(),
        "questions": set(),
        "ids": set(),
    }
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                deck = str(row.get("deck_name") or "")
                question = extract_question(row)
                if deck and question:
                    exclusions["deck_question_keys"].add((deck, normalize_question(question)))
                if question:
                    exclusions["questions"].add(normalize_question(question))
                for key in ("sample_id", "qa_id", "id"):
                    value = row.get(key)
                    if value is not None and str(value).strip():
                        exclusions["ids"].add(str(value).strip())
    return exclusions


def extract_question(row: dict[str, Any]) -> str:
    question = str(row.get("question") or row.get("query") or "").strip()
    if question:
        return question
    texts = []
    for turn in row.get("conversations") or []:
        if turn.get("from") == "human":
            texts.append(str(turn.get("value") or ""))
    text = "\n".join(texts)
    patterns = [
        r"Original question:\s*\n(.+?)(?:\n\s*\n|\nPage image:|\nInstruction:|\nInput:|$)",
        r"Question:\s*\n?(.+?)(?:\n\s*\n|\nSuccessful search history|\nFailed search history|\nCurrent evidence_state|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            value = re.sub(r"\n.*", "", match.group(1).strip()).strip()
            if value and value != "<image>":
                return value
    return ""


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", str(question or "").lower().strip())

def collect_candidates(
    *,
    parquet_root: Path,
    split: str,
    exclude: dict[str, set[Any]],
    max_images: int,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    candidates: list[dict[str, Any]] = []
    columns = ["deck_name", "qa_id", "question", "answer", "evidence_pages"]
    for parquet_path in sorted(parquet_root.glob(f"{split}-*.parquet")):
        table = pq.read_table(parquet_path, columns=columns)
        for row_index, row in enumerate(table.to_pylist()):
            deck = str(row.get("deck_name") or "")
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            pages = normalize_pages(row.get("evidence_pages"), max_images=max_images)
            if not deck or not question or not answer or not pages:
                continue
            qa_id = str(row.get("qa_id") or "").strip()
            normalized_question = normalize_question(question)
            if qa_id and qa_id in exclude["ids"]:
                continue
            if normalized_question in exclude["questions"]:
                continue
            if (deck, normalized_question) in exclude["deck_question_keys"]:
                continue
            candidates.append(
                {
                    "source_parquet": str(parquet_path),
                    "row_index": row_index,
                    "deck_name": deck,
                    "qa_id": row.get("qa_id"),
                    "question": question,
                    "answer": answer,
                    "evidence_pages": pages,
                    "hop_type": "single" if len(pages) == 1 else "multi",
                }
            )
    return candidates


def normalize_pages(value: Any, *, max_images: int) -> list[int]:
    pages: list[int] = []
    for page in value or []:
        try:
            page_num = int(page)
        except (TypeError, ValueError):
            continue
        if 1 <= page_num <= 20 and page_num not in pages:
            pages.append(page_num)
        if len(pages) >= max_images:
            break
    return pages


def select_balanced(
    *,
    candidates: list[dict[str, Any]],
    train_single: int,
    train_multi: int,
    val_single: int,
    val_multi: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_hop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_hop[item["hop_type"]].append(item)
    for items in by_hop.values():
        rng.shuffle(items)

    needs = {
        "single": train_single + val_single,
        "multi": train_multi + val_multi,
    }
    for hop_type, needed in needs.items():
        available = len(by_hop[hop_type])
        if available < needed:
            raise ValueError(f"not enough {hop_type} candidates: need {needed}, got {available}")

    train = by_hop["single"][:train_single] + by_hop["multi"][:train_multi]
    val = (
        by_hop["single"][train_single : train_single + val_single]
        + by_hop["multi"][train_multi : train_multi + val_multi]
    )
    rng.shuffle(train)
    rng.shuffle(val)
    return {"train": train, "val": val}


def materialize_rows(
    selected: list[dict[str, Any]],
    *,
    max_images: int,
    data_source: str,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    selected_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        selected_by_file[item["source_parquet"]].append(item)

    rows: list[dict[str, Any]] = []
    for parquet_path, items in selected_by_file.items():
        needed_pages = sorted({page for item in items for page in item["evidence_pages"]})
        columns = [
            "deck_name",
            "qa_id",
            "question",
            "answer",
            "evidence_pages",
            *[f"page_{page}" for page in needed_pages],
        ]
        table = pq.read_table(parquet_path, columns=columns)
        rows_by_index = {item["row_index"]: item for item in items}
        for row_index, source_row in enumerate(table.to_pylist()):
            item = rows_by_index.get(row_index)
            if item is None:
                continue
            images = []
            page_labels = []
            for page_num in item["evidence_pages"][:max_images]:
                image_obj = source_row.get(f"page_{page_num}")
                image_bytes = extract_image_bytes(image_obj)
                if not image_bytes:
                    continue
                label = f"{item['deck_name']}/page_{page_num:02d}"
                images.append({"bytes": image_bytes, "path": f"{label}.jpg"})
                page_labels.append(label)
            if len(images) != len(item["evidence_pages"][:max_images]):
                continue
            prompt = (
                "Answer the visual document question using only the attached page image(s). "
                "Return only the concise final answer, without citations or extra explanation.\n\n"
                f"Question: {item['question']}"
            )
            rows.append(
                {
                    "data_source": data_source,
                    "prompt": [{"role": "user", "content": prompt}],
                    "images": images,
                    "ability": "visual_document_qa",
                    "reward_model": {"style": "rule", "ground_truth": item["answer"]},
                    "extra_info": {
                        "split": "train",
                        "source_parquet": Path(parquet_path).name,
                        "row_index": item["row_index"],
                        "qa_id": item["qa_id"],
                        "deck_name": item["deck_name"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "evidence_pages": item["evidence_pages"],
                        "page_labels": page_labels,
                        "hop_type": item["hop_type"],
                    },
                }
            )
    return rows


def extract_image_bytes(value: Any) -> bytes | None:
    if isinstance(value, dict):
        raw = value.get("bytes")
        return raw if isinstance(raw, bytes) else None
    return value if isinstance(value, bytes) else None


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def count_by_hop(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        hop_type = row.get("hop_type")
        if not hop_type:
            hop_type = (row.get("extra_info") or {}).get("hop_type")
        counts[str(hop_type)] += 1
    return dict(sorted(counts.items()))


def count_overlap(rows: list[dict[str, Any]], exclude_keys: set[tuple[str, str]]) -> int:
    total = 0
    for row in rows:
        info = row.get("extra_info") or {}
        key = (str(info.get("deck_name") or ""), str(info.get("question") or "").strip())
        if key in exclude_keys:
            total += 1
    return total


if __name__ == "__main__":
    main()
