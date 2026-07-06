#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

from src.prompt_serialization import build_agent_human_turn
from train_sft_qwen_vl import TrajectorySftDataset


DEFAULT_MIMO25_RUN_ROOT = (
    "outputs/sft_trajectories/mimo25_train_300multi_shortest_900single_calllevel_20260527"
)
DEFAULT_TRAIN_FILE = f"{DEFAULT_MIMO25_RUN_ROOT}/kept_selected_sft_calls.jsonl"
DEFAULT_IMAGE_ROOT = "data/corpora/slidevqa_sft/pages"
DEFAULT_OUTPUT_FILE = "outputs/sft_qwenvl/mimo25_300multi_shortest_900single_calllevel_sft.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert generated agent trajectories to Qwen3-VL official SFT JSONL."
    )
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--image-root", action="append", default=[DEFAULT_IMAGE_ROOT])
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument(
        "--packing",
        choices=["call", "trajectory"],
        default="call",
        help="call keeps one row per model call; trajectory groups calls from the same sample into one multi-turn row.",
    )
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument("--include-text-only", action="store_true", default=True)
    parser.add_argument("--no-text-only", dest="include_text_only", action="store_false")
    args = parser.parse_args()

    dataset = TrajectorySftDataset(
        train_file=Path(args.train_file),
        image_roots=[Path(p) for p in args.image_root],
        limit=args.limit,
        shuffle=args.shuffle if args.packing == "call" else False,
        seed=args.seed,
        skip_missing_images=args.skip_missing_images,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_counts: dict[int, int] = {}
    skipped_text_only = 0
    if args.packing == "call":
        rows = []
        step_counts: dict[str, int] = {}
        for idx, example in enumerate(iter_dataset(dataset)):
            if not example.image_paths and not args.include_text_only:
                skipped_text_only += 1
                continue
            rows.append(to_qwenvl_call_row(example, idx))
            step_counts[example.step] = step_counts.get(example.step, 0) + 1
            image_counts[len(example.image_paths)] = image_counts.get(len(example.image_paths), 0) + 1
    else:
        rows, skipped_text_only = to_qwenvl_trajectory_rows(
            iter_dataset(dataset),
            include_text_only=args.include_text_only,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        step_counts = dict(dataset.step_counts)
        for row in rows:
            image_counts[len(row.get("image") or [])] = image_counts.get(len(row.get("image") or []), 0) + 1

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_summary": dataset.summary(),
                "output_file": str(output_path),
                "packing": args.packing,
                "qwenvl_examples": len(rows),
                "step_counts": step_counts,
                "image_counts": image_counts,
                "skipped_text_only": skipped_text_only,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def iter_dataset(dataset: Any) -> list[Any]:
    return [dataset[idx] for idx in range(len(dataset))]


def to_qwenvl_call_row(example: Any, idx: int) -> dict[str, Any]:
    images = [str(Path(path).resolve()) for path in example.image_paths]

    row: dict[str, Any] = {
        "id": f"{example.sample_id}:{example.step}:{idx}",
        "sample_id": example.sample_id,
        "step": example.step,
        "conversations": [
            {
                "from": "human",
                "value": build_human_value(
                    example.system,
                    example.user,
                    images,
                    call_type=example.step,
                ),
            },
            {"from": "gpt", "value": example.target},
        ],
    }
    if images:
        row["image"] = images
    return row


def to_qwenvl_trajectory_rows(
    examples: list[Any],
    *,
    include_text_only: bool,
    shuffle: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    grouped: "OrderedDict[str, list[Any]]" = OrderedDict()
    for example in examples:
        grouped.setdefault(str(example.sample_id), []).append(example)

    rows: list[dict[str, Any]] = []
    skipped_text_only = 0
    for sample_id, items in grouped.items():
        images: list[str] = []
        conversations: list[dict[str, str]] = []
        step_counts: dict[str, int] = {}
        for item in items:
            item_images = [str(Path(path).resolve()) for path in item.image_paths]
            images.extend(item_images)
            step_counts[item.step] = step_counts.get(item.step, 0) + 1
            conversations.append(
                {
                    "from": "human",
                    "value": build_human_value(
                        item.system,
                        item.user,
                        item_images,
                        call_type=item.step,
                    ),
                }
            )
            conversations.append({"from": "gpt", "value": item.target})
        if not images and not include_text_only:
            skipped_text_only += 1
            continue
        row: dict[str, Any] = {
            "id": sample_id,
            "sample_id": sample_id,
            "packing": "trajectory",
            "step_counts": step_counts,
            "conversations": conversations,
        }
        if images:
            row["image"] = images
        rows.append(row)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    return rows, skipped_text_only


def build_human_value(
    system: str,
    user: str,
    images: list[str],
    *,
    call_type: str | None,
) -> str:
    image_placeholders = "\n".join("<image>" for _ in images)
    parts: list[str] = []
    if image_placeholders:
        parts.append(image_placeholders)
    parts.append(build_agent_human_turn(system, user, call_type=call_type))
    return "\n\n".join(part for part in parts if part)


if __name__ == "__main__":
    main()
