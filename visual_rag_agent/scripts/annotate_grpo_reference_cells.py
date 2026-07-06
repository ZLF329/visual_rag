#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_relevance_warmup_cells import (
    CELL_LABEL_USER,
    call_openai_compatible_api,
    cells_to_labels,
    normalize_model_name,
    parse_useful_cells,
)
from src.image_utils import add_coordinate_grid, resize_to_pixel_count

DEFAULT_INPUT_DIR = "data/grpo/visor_slidevqa_balanced_800"
DEFAULT_OUTPUT = "outputs/rl_annotations/grpo800_kimi_k26_reference_cells.jsonl"
DEFAULT_AUDIT_ROOT = "outputs/rl_annotations/grpo800_kimi_k26_reference_cells_audit"
DEFAULT_MODEL = "kimi-k2.6"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass(frozen=True)
class GrpoCellTask:
    row_id: str
    split: str
    row_index: int
    qa_id: str
    deck_name: str
    question: str
    answer: str
    hop_type: str
    page_number: int
    page_label: str
    image_bytes: bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate clean GRPO reference pages with useful 8x8 cells via an OpenAI-compatible vision API.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-root", default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--image-pixels", type=int, default=900_000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("QWEN_BASE_URL") or os.environ.get("DASHSCOPE_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    if not args.api_key.strip():
        raise SystemExit("Missing --api-key or QWEN_API_KEY/DASHSCOPE_API_KEY/OPENAI_API_KEY")

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_file)
    audit_root = Path(args.audit_root)
    grid_root = audit_root / "grid_images"
    raw_path = audit_root / "raw_annotations.jsonl"
    error_path = audit_root / "errors.jsonl"
    summary_path = output_path.with_suffix(".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    grid_root.mkdir(parents=True, exist_ok=True)

    completed_ids = load_existing_ids(output_path)
    tasks = collect_tasks(input_dir=input_dir, splits=args.splits)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    tasks = [task for task in tasks if task.row_id not in completed_ids]

    model = normalize_model_name(args.model, base_url=args.base_url)
    counters: dict[str, Any] = {
        "model": model,
        "base_url": args.base_url,
        "input_dir": str(input_dir),
        "splits": args.splits,
        "requested_limit": args.limit,
        "queued_tasks": len(tasks),
        "skipped_existing": len(completed_ids),
        "completed": 0,
        "nonempty": 0,
        "empty": 0,
        "api_errors": 0,
        "parse_errors": 0,
        "total_cells": 0,
        "split_counts": {},
        "hop_counts": {},
        "image_pixels": args.image_pixels,
    }

    writer_lock = threading.Lock()
    session_local = threading.local()

    def get_session() -> requests.Session:
        session = getattr(session_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"})
            session_local.session = session
        return session

    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(
                process_task,
                task=task,
                session_getter=get_session,
                base_url=args.base_url,
                model=model,
                image_pixels=args.image_pixels,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                enable_thinking=args.enable_thinking,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
                grid_root=grid_root,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                counters["api_errors"] += 1
                append_jsonl(error_path, {"row_id": task.row_id, "error": repr(exc)}, writer_lock)
                continue
            if result["status"] != "ok":
                key = "parse_errors" if result["status"] == "parse_error" else "api_errors"
                counters[key] += 1
                append_jsonl(error_path, result, writer_lock)
                continue
            row = result["row"]
            cells = row["useful_cells"]
            counters["completed"] += 1
            counters["total_cells"] += len(cells)
            counters["nonempty" if cells else "empty"] += 1
            counters["split_counts"][task.split] = counters["split_counts"].get(task.split, 0) + 1
            counters["hop_counts"][task.hop_type] = counters["hop_counts"].get(task.hop_type, 0) + 1
            append_jsonl(output_path, row, writer_lock)
            append_jsonl(
                raw_path,
                {
                    "row_id": task.row_id,
                    "split": task.split,
                    "qa_id": task.qa_id,
                    "page_label": task.page_label,
                    "grid_image_path": result["grid_image_path"],
                    "raw_response_text": result["raw_response_text"],
                    "useful_cells": cells,
                },
                writer_lock,
            )

    counters["elapsed_sec"] = round(time.time() - start, 2)
    counters["output_file"] = str(output_path)
    counters["audit_root"] = str(audit_root)
    counters["avg_cells_per_completed"] = counters["total_cells"] / counters["completed"] if counters["completed"] else 0.0
    summary_path.write_text(json.dumps(counters, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counters, ensure_ascii=False, indent=2))


def collect_tasks(*, input_dir: Path, splits: list[str]) -> list[GrpoCellTask]:
    import pandas as pd

    tasks: list[GrpoCellTask] = []
    seen: set[str] = set()
    for split in splits:
        df = pd.read_parquet(input_dir / f"{split}.parquet")
        for row_index, row in df.iterrows():
            extra = dict(row.get("extra_info") or {})
            qa_id = str(extra.get("qa_id") or f"{split}_{row_index}")
            deck_name = str(extra.get("deck_name") or "")
            question = str(extra.get("question") or "").strip()
            answer = str(extra.get("answer") or "").strip()
            hop_type = str(extra.get("hop_type") or "unknown")
            labels = [str(x) for x in as_list(extra.get("page_labels"))]
            pages = normalize_pages(extra.get("evidence_pages"))
            images = as_list(row.get("images"))
            for pos, page_label in enumerate(labels):
                page_number = pages[pos] if pos < len(pages) else page_number_from_label(page_label)
                image_obj = images[pos] if pos < len(images) else None
                image_bytes = extract_image_bytes(image_obj)
                if not image_bytes:
                    continue
                row_id = f"grpo_ref_cells:{split}:{qa_id}:{page_label}"
                if row_id in seen:
                    continue
                seen.add(row_id)
                tasks.append(
                    GrpoCellTask(
                        row_id=row_id,
                        split=split,
                        row_index=int(row_index),
                        qa_id=qa_id,
                        deck_name=deck_name,
                        question=question,
                        answer=answer,
                        hop_type=hop_type,
                        page_number=page_number,
                        page_label=page_label,
                        image_bytes=image_bytes,
                    )
                )
    return tasks


def process_task(*, task: GrpoCellTask, session_getter, base_url: str, model: str, image_pixels: int, temperature: float | None, max_tokens: int, enable_thinking: bool, timeout_sec: int, max_retries: int, grid_root: Path) -> dict[str, Any]:
    grid_image_path = materialize_grid_image(task, grid_root=grid_root, image_pixels=image_pixels)
    image_data_url = grid_image_data_url(grid_image_path)
    prompt = CELL_LABEL_USER.format(question=task.question or "(empty question)", answer=task.answer or "(unknown)")
    raw_text = ""
    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_text = call_openai_compatible_api(
                session=session_getter(),
                base_url=base_url,
                model=model,
                prompt=prompt,
                image_data_url=image_data_url,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                timeout_sec=timeout_sec,
            )
            cells = parse_useful_cells(raw_text)
            row = {
                "id": task.row_id,
                "split": task.split,
                "row_index": task.row_index,
                "qa_id": task.qa_id,
                "deck_name": task.deck_name,
                "question": task.question,
                "answer": task.answer,
                "hop_type": task.hop_type,
                "page_number": task.page_number,
                "page_label": task.page_label,
                "useful_cells": cells,
                "cell_labels": cells_to_labels(cells),
                "cell_loss_mask": 1.0,
                "annotation_model": model,
                "annotation_source": "qwenapi_kimi_k26_grpo_reference_cell_only",
            }
            return {"status": "ok", "row": row, "grid_image_path": str(grid_image_path), "raw_response_text": raw_text}
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            if attempt >= max_retries:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    status = "parse_error" if last_error and ("json" in last_error.lower() or "cell" in last_error.lower()) else "api_error"
    return {"status": status, "row_id": task.row_id, "split": task.split, "qa_id": task.qa_id, "page_label": task.page_label, "raw_response_text": raw_text, "error": last_error}


def materialize_grid_image(task: GrpoCellTask, *, grid_root: Path, image_pixels: int) -> Path:
    safe_label = task.page_label.replace("/", "__")
    out_path = grid_root / task.split / f"{task.qa_id}__{safe_label}.png"
    if out_path.exists():
        return out_path
    with Image.open(io.BytesIO(task.image_bytes)) as image:
        resized = resize_to_pixel_count(image.convert("RGB"), image_pixels, allow_upscale=False)
        gridded = add_coordinate_grid(resized)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gridded.save(out_path)
    return out_path


def grid_image_data_url(path: Path) -> str:
    with Image.open(path) as image:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def extract_image_bytes(value: Any) -> bytes:
    if isinstance(value, dict) and value.get("bytes") is not None:
        return bytes(value["bytes"])
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return b""


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


def page_number_from_label(label: str) -> int:
    try:
        return int(label.rsplit("page_", 1)[1])
    except Exception:
        return -1


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = row.get("id")
            if isinstance(row_id, str) and row_id:
                ids.add(row_id)
    return ids


def append_jsonl(path: Path, obj: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
