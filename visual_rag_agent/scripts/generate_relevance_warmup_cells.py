#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from src.image_utils import add_coordinate_grid, resize_to_pixel_count

COLS = "ABCDEFGH"
CELL_RE = re.compile(r"^[A-H][1-8]$")

DEFAULT_INPUTS = [
    "data/corpora/slidevqa_train_balanced_2000/train_single.jsonl",
    "data/corpora/slidevqa_train_balanced_2000/train_multi.jsonl",
]
DEFAULT_OUTPUT = "outputs/sft_qwenvl/relevance_head_warmup_kimi_k26_extra_cells.jsonl"
DEFAULT_AUDIT_ROOT = "outputs/sft_qwenvl/relevance_head_warmup_kimi_k26_audit"
DEFAULT_IMAGE_PIXELS = 900_000
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "kimi-k2.6"
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

WARMUP_ANALYSE_INSTRUCTION = """Analyse this page for the original question.
Return JSON only:
{
  "think": "...",
  "summary": "...",
  "observed_evidence": "...",
  "judge": "yes" | "partial" | "no"
}

Use "yes" only when this page contains explicit evidence for the full answer.
Use "partial" when it contributes useful evidence but does not fully answer.
Use "no" when it does not help. Keep summary to one concise sentence."""

WARMUP_ANALYSE_USER = """Agent call type: analyse

Original question:
{original_query}

Page image:
<image>

Instruction:
{instruction}"""

CELL_LABEL_SYSTEM = """You label useful 8x8 overlaid grid cells on slide pages.
Return JSON only. Do not include markdown fences or commentary."""

CELL_LABEL_USER = """Task: identify the overlaid 8x8 grid cells on this page that contain evidence useful for answering the question.

Original question:
{question}

Known correct answer (for localization only):
{answer}

Rules:
- The attached page image already has an overlaid 8x8 grid labeled A1 to H8.
- Return only cells that visibly contain useful evidence for answering the question.
- If the page does not visibly support the answer, return an empty list.
- For charts, tables, or dense layouts, include the broader cells that cover the relevant header, label, and value.
- Every returned cell must match ^[A-H][1-8]$ exactly.

Return JSON only with this schema:
{{
  "useful_cells": ["A1", "B1"]
}}"""


@dataclass(frozen=True)
class Task:
    row_id: str
    sample_id: int | str
    hop_type: str
    input_file: str
    source_row_index: int | None
    question: str
    answer: str
    page_number: int
    source_image_path: str
    source_image_rel: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate extra warmup rows for the relevance head by asking a "
            "vision API model to label useful 8x8 cells on evidence pages."
        )
    )
    parser.add_argument("--input-files", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-root", default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of page rows to generate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle candidate pages before sampling.")
    parser.add_argument("--image-pixels", type=int, default=DEFAULT_IMAGE_PIXELS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--base-url",
        default=(
            os.environ.get("QWEN_BASE_URL")
            or os.environ.get("DASHSCOPE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_DASHSCOPE_BASE_URL
        ),
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("QWEN_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable DashScope Kimi thinking mode. Default is off for cheaper, cleaner cell labeling.",
    )
    parser.add_argument(
        "--page-field",
        choices=["evidence_pages", "reference_pages", "auto"],
        default="auto",
        help="Which page index field to use from the source corpus.",
    )
    parser.add_argument(
        "--nonempty-judge",
        choices=["partial", "yes"],
        default="partial",
        help="Judge label to write for rows with non-empty useful_cells. LM loss stays 0 either way.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_url.strip():
        raise SystemExit("Missing --base-url (or QWEN_BASE_URL / OPENAI_BASE_URL).")
    if not args.api_key.strip():
        raise SystemExit(
            "Missing --api-key (or QWEN_API_KEY / DASHSCOPE_API_KEY / OPENAI_API_KEY)."
        )

    request_model = normalize_model_name(args.model, base_url=args.base_url)

    output_path = Path(args.output_file)
    audit_root = Path(args.audit_root)
    grid_root = audit_root / "grid_images"
    raw_output_path = audit_root / "raw_annotations.jsonl"
    error_output_path = audit_root / "errors.jsonl"
    summary_path = output_path.with_suffix(".summary.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    grid_root.mkdir(parents=True, exist_ok=True)

    completed_ids = load_existing_ids(output_path)
    tasks = collect_tasks(
        input_files=[Path(p) for p in args.input_files],
        page_field=args.page_field,
    )
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    tasks = [task for task in tasks if task.row_id not in completed_ids]

    writer_lock = threading.Lock()
    session_local = threading.local()
    counters: dict[str, Any] = {
        "model": request_model,
        "base_url": args.base_url,
        "input_files": [str(p) for p in args.input_files],
        "requested_limit": args.limit,
        "queued_tasks": len(tasks),
        "skipped_existing": len(completed_ids),
        "completed": 0,
        "nonempty": 0,
        "empty": 0,
        "api_errors": 0,
        "parse_errors": 0,
        "total_cells": 0,
        "hop_counts": {},
        "image_pixels": args.image_pixels,
    }

    def get_session() -> requests.Session:
        session = getattr(session_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Authorization": f"Bearer {args.api_key}",
                    "Content-Type": "application/json",
                }
            )
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
                model=request_model,
                image_pixels=args.image_pixels,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                enable_thinking=args.enable_thinking,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
                grid_root=grid_root,
                nonempty_judge=args.nonempty_judge,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                counters["api_errors"] += 1
                append_jsonl(
                    error_output_path,
                    {
                        "row_id": task.row_id,
                        "sample_id": task.sample_id,
                        "source_image_path": task.source_image_path,
                        "error": repr(exc),
                    },
                    writer_lock,
                )
                continue

            if result["status"] != "ok":
                key = "parse_errors" if result["status"] == "parse_error" else "api_errors"
                counters[key] += 1
                append_jsonl(error_output_path, result, writer_lock)
                continue

            row = result["row"]
            useful_cells = row["useful_cells"]
            counters["completed"] += 1
            counters["total_cells"] += len(useful_cells)
            if useful_cells:
                counters["nonempty"] += 1
            else:
                counters["empty"] += 1
            counters["hop_counts"][task.hop_type] = counters["hop_counts"].get(task.hop_type, 0) + 1

            append_jsonl(output_path, row, writer_lock)
            append_jsonl(
                raw_output_path,
                {
                    "row_id": task.row_id,
                    "sample_id": task.sample_id,
                    "source_image_path": task.source_image_path,
                    "grid_image_path": result["grid_image_path"],
                    "raw_response_text": result["raw_response_text"],
                    "useful_cells": useful_cells,
                },
                writer_lock,
            )

    elapsed = round(time.time() - start, 2)
    counters["elapsed_sec"] = elapsed
    counters["output_file"] = str(output_path)
    counters["audit_root"] = str(audit_root)
    counters["avg_cells_per_completed"] = (
        counters["total_cells"] / counters["completed"] if counters["completed"] else 0.0
    )
    summary_path.write_text(json.dumps(counters, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counters, ensure_ascii=False, indent=2))


def collect_tasks(*, input_files: list[Path], page_field: str) -> list[Task]:
    tasks: list[Task] = []
    seen: set[str] = set()
    for input_path in input_files:
        corpus_root = input_path.resolve().parent
        pages_root = corpus_root / "pages"
        with input_path.open(encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                record = json.loads(line)
                candidate_pages = page_numbers_for_record(record, page_field=page_field)
                page_images = list(record.get("page_images") or [])
                for page_number in candidate_pages:
                    if page_number <= 0 or page_number > len(page_images):
                        continue
                    rel_path = str(page_images[page_number - 1])
                    source_image_path = str((pages_root / rel_path).resolve())
                    row_id = f"warmup_cells:{Path(input_path).stem}:{record.get('id')}:{page_number}"
                    if row_id in seen:
                        continue
                    seen.add(row_id)
                    tasks.append(
                        Task(
                            row_id=row_id,
                            sample_id=record.get("id"),
                            hop_type=str(record.get("hop_type") or "unknown"),
                            input_file=str(input_path),
                            source_row_index=record.get("source_row_index"),
                            question=str(record.get("question") or record.get("query") or "").strip(),
                            answer=str(record.get("answer") or "").strip(),
                            page_number=page_number,
                            source_image_path=source_image_path,
                            source_image_rel=rel_path,
                        )
                    )
    return tasks


def page_numbers_for_record(record: dict[str, Any], *, page_field: str) -> list[int]:
    if page_field == "evidence_pages":
        pages = record.get("evidence_pages") or []
    elif page_field == "reference_pages":
        pages = record.get("reference_pages") or []
    else:
        pages = record.get("evidence_pages") or record.get("reference_pages") or []
    out: list[int] = []
    for value in pages:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number not in out:
            out.append(number)
    return out


def process_task(
    *,
    task: Task,
    session_getter,
    base_url: str,
    model: str,
    image_pixels: int,
    temperature: float | None,
    max_tokens: int,
    enable_thinking: bool,
    timeout_sec: int,
    max_retries: int,
    grid_root: Path,
    nonempty_judge: str,
) -> dict[str, Any]:
    grid_image_path = materialize_grid_image(task, grid_root=grid_root, image_pixels=image_pixels)
    image_data_url = grid_image_data_url(grid_image_path)
    prompt = CELL_LABEL_USER.format(
        question=task.question or "(empty question)",
        answer=task.answer or "(unknown)",
    )

    last_error: str | None = None
    raw_text = ""
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
            useful_cells = parse_useful_cells(raw_text)
            row = build_warmup_row(
                task=task,
                useful_cells=useful_cells,
                nonempty_judge=nonempty_judge,
                model=model,
            )
            return {
                "status": "ok",
                "row": row,
                "grid_image_path": str(grid_image_path),
                "raw_response_text": raw_text,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            if attempt >= max_retries:
                break
            time.sleep(min(2 ** (attempt - 1), 8))

    status = "parse_error" if last_error and ("json" in last_error.lower() or "cell" in last_error.lower()) else "api_error"
    return {
        "status": status,
        "row_id": task.row_id,
        "sample_id": task.sample_id,
        "source_image_path": task.source_image_path,
        "grid_image_path": str(grid_image_path),
        "raw_response_text": raw_text,
        "error": last_error,
    }


def build_warmup_row(
    *,
    task: Task,
    useful_cells: list[str],
    nonempty_judge: str,
    model: str,
) -> dict[str, Any]:
    judge = nonempty_judge if useful_cells else "no"
    target = {
        "think": "",
        "summary": "",
        "observed_evidence": "",
        "judge": judge,
    }
    user_turn = WARMUP_ANALYSE_USER.format(
        original_query=task.question,
        instruction=WARMUP_ANALYSE_INSTRUCTION,
    )
    source_image_path = str(Path(task.source_image_path).resolve())
    return {
        "id": task.row_id,
        "sample_id": task.sample_id,
        "source_hop_type": task.hop_type,
        "step": "analyse",
        "judge": judge,
        "search_query": None,
        "image": [source_image_path],
        "source_image_path": source_image_path,
        "cell_labels": cells_to_labels(useful_cells),
        "useful_cells": useful_cells,
        "cell_loss_mask": 1.0,
        "lm_loss_weight": 0.0,
        "cell_loss_weight": 1.0,
        "annotation_model": model,
        "annotation_source": "qwenapi_cell_only",
        "source_input_file": task.input_file,
        "source_row_index": task.source_row_index,
        "source_page_number": task.page_number,
        "conversations": [
            {"from": "human", "value": user_turn},
            {"from": "gpt", "value": json.dumps(target, ensure_ascii=False)},
        ],
    }


def materialize_grid_image(task: Task, *, grid_root: Path, image_pixels: int) -> Path:
    safe_rel = task.source_image_rel.replace("/", "__")
    out_path = grid_root / f"{Path(task.input_file).stem}__{task.sample_id}__p{task.page_number:02d}__{safe_rel}"
    out_path = out_path.with_suffix(".png")
    if out_path.exists():
        return out_path

    with Image.open(task.source_image_path) as image:
        resized = resize_to_pixel_count(image.convert("RGB"), image_pixels, allow_upscale=False)
        gridded = add_coordinate_grid(resized)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gridded.save(out_path)
    return out_path


def grid_image_data_url(grid_image_path: Path) -> str:
    with Image.open(grid_image_path) as image:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def call_openai_compatible_api(
    *,
    session: requests.Session,
    base_url: str,
    model: str,
    prompt: str,
    image_data_url: str,
    temperature: float | None,
    max_tokens: int,
    enable_thinking: bool,
    timeout_sec: int,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": CELL_LABEL_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    payload["enable_thinking"] = bool(enable_thinking)
    response = session.post(url, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"no choices in response: {data}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        joined = "".join(text_parts).strip()
        if joined:
            return joined
    raise ValueError(f"missing text content in response: {data}")


def parse_useful_cells(raw_text: str) -> list[str]:
    obj: Any
    try:
        obj = json.loads(raw_text)
    except json.JSONDecodeError:
        obj = json.loads(extract_json_fragment(raw_text))

    if isinstance(obj, list):
        cells = obj
    elif isinstance(obj, dict):
        cells = obj.get("useful_cells")
    else:
        raise ValueError(f"unsupported JSON type for useful_cells: {type(obj).__name__}")

    if cells is None:
        raise ValueError(f"useful_cells missing in response: {raw_text}")
    if not isinstance(cells, list):
        raise ValueError(f"useful_cells must be a list: {raw_text}")

    normalized: list[str] = []
    for cell in cells:
        token = str(cell).strip().upper()
        if not CELL_RE.fullmatch(token):
            raise ValueError(f"invalid cell label {token!r} in response: {raw_text}")
        if token not in normalized:
            normalized.append(token)
    normalized.sort(key=cell_sort_key)
    return normalized


def extract_json_fragment(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found: {text}")
    return text[start : end + 1]


def normalize_model_name(model: str, *, base_url: str) -> str:
    normalized = model.strip()
    return normalized


def cell_sort_key(cell: str) -> tuple[int, int]:
    return (int(cell[1]) - 1, COLS.index(cell[0]))


def cells_to_labels(cells: list[str]) -> list[float]:
    labels = [0.0] * 64
    for cell in cells:
        col = COLS.index(cell[0])
        row = int(cell[1]) - 1
        labels[row * 8 + col] = 1.0
    return labels


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
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
