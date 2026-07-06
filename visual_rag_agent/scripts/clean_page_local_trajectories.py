#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

DEFAULT_INPUT_DIR = Path("outputs/sft_trajectories/gpt5mini_20260525_153719")
DEFAULT_FILES = (
    "kept_sft_trajectories.jsonl",
    "raw_trajectories.jsonl",
    "rejected_trajectories.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean page-local SFT trajectories.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    parser.add_argument("--suffix", default=".page_local_clean")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    summaries = []
    for name in args.files:
        src = input_dir / name
        if not src.exists():
            summaries.append({"input": str(src), "missing": True})
            continue
        dst = src.with_name(src.stem + args.suffix + src.suffix)
        summary = clean_file(src, dst)
        summaries.append(summary)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def clean_file(src: Path, dst: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "input": str(src),
        "output": str(dst),
        "records": 0,
        "error_records": 0,
        "step_counts": {},
        "input_step_counts": {},
        "discarded_step_counts": {},
        "judge_counts": {},
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            record = json.loads(line)
            stats["records"] += 1
            if not isinstance(record.get("trace"), list):
                stats["error_records"] += 1
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            clean_record(record, stats)
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return stats


def clean_record(record: dict[str, Any], stats: dict[str, Any]) -> None:
    reference_labels = normalize_label_set(record.get("reference_page_labels"))
    retrieved_labels: list[str] = []
    found_labels: set[str] = set()
    consolidated_summaries: list[str] = []
    observations: list[dict[str, Any]] = []
    confirmed: list[dict[str, Any]] = []
    partials: deque[dict[str, Any]] = deque(maxlen=2)
    cleaned_trace: list[dict[str, Any]] = []
    current_query = ""
    image_dir = (record.get("memory") or {}).get("image_dir")

    for step in record.get("trace", []):
        kind = step.get("step")
        stats["input_step_counts"][kind] = stats["input_step_counts"].get(kind, 0) + 1

        if kind == "decide":
            result = clean_decide_payload(step.get("result") or parse_json(step.get("raw_text")) or {})
            step["result"] = result
            step["raw_text"] = dump_json(result)
            record_output_step(stats, kind)
            cleaned_trace.append(step)
            continue

        if kind == "search":
            current_query = str(step.get("query") or "")
            for label in page_labels_from_pages(step.get("pages") or []):
                retrieved_labels.append(label)
            record_output_step(stats, kind)
            cleaned_trace.append(step)
            continue

        if kind == "analyse":
            page_label = str(step.get("page") or "")
            if page_label:
                retrieved_labels.append(page_label)
            result = clean_analyse_payload(
                step.get("result") or parse_json(step.get("raw_text")) or {},
                page_label=page_label,
                reference_labels=reference_labels,
            )
            step["result"] = result
            step["raw_text"] = dump_json(result)
            judge = result["judge"]
            stats["judge_counts"][judge] = stats["judge_counts"].get(judge, 0) + 1
            if page_label in reference_labels:
                found_labels.add(page_label)
            summary = str(result.get("summary") or "").strip()
            query = str(step.get("query") or current_query or "").strip()
            if summary:
                consolidated_summaries.append(f"[query: {query}] {summary}" if query else summary)
            observation = {
                "iter": step.get("iter"),
                "search_query": query,
                "page_label": page_label,
                "decision": judge,
                "useful_cells": result.get("useful_cells") or [],
                "summary": summary,
            }
            observations.append(observation)
            if judge in {"yes", "partial"}:
                image_ref = {"page_label": page_label, "path": first_retained_path(step), "pixel_budget": 400000 if judge == "yes" else 200000}
                if judge == "yes":
                    confirmed.append({"image_ref": image_ref, "found_via_query": query, "useful_cells": result.get("useful_cells") or []})
                else:
                    partials.append({"image_ref": image_ref, "found_via_query": query, "found_at_iter": step.get("iter"), "useful_cells": result.get("useful_cells") or []})
            record_output_step(stats, kind)
            cleaned_trace.append(step)
            continue

    retrieved_unique = unique_preserve_order(retrieved_labels)
    found_sorted = sorted(found_labels)
    missing_sorted = sorted(reference_labels - found_labels)
    record["retrieved_page_labels"] = retrieved_unique
    record["found_reference_page_labels"] = found_sorted
    record["missing_reference_page_labels"] = missing_sorted
    record["trace"] = cleaned_trace
    record["memory"] = {
        "original_query": record.get("question") or "",
        "confirmed": confirmed,
        "partials": list(partials),
        "warnings": [],
        "observations": observations,
        "consolidated_summary": "\n".join(consolidated_summaries),
        "consolidated_summaries": consolidated_summaries,
        "iter": max([int(s.get("iter") or 0) for s in cleaned_trace] or [0]),
        "image_dir": image_dir,
    }
    record["sft_messages"] = build_sft_messages(record)


def record_output_step(stats: dict[str, Any], kind: Any) -> None:
    stats["step_counts"][kind] = stats["step_counts"].get(kind, 0) + 1


def clean_decide_payload(payload: dict[str, Any]) -> dict[str, Any]:
    action = payload.get("action") if payload.get("action") in {"search", "answer"} else "search"
    content = payload.get("content")
    if action == "search":
        content = str(content or payload.get("search_query") or payload.get("answer") or "").strip()
    else:
        content = str(content or payload.get("answer") or payload.get("search_query") or "").strip()
    return {
        "think": str(payload.get("think") or "").strip(),
        "action": action,
        "content": content,
    }


def clean_analyse_payload(payload: dict[str, Any], *, page_label: str, reference_labels: set[str]) -> dict[str, Any]:
    if page_label not in reference_labels:
        judge = "no"
    elif len(reference_labels) == 1:
        judge = "yes"
    else:
        judge = "partial"
    useful_cells = clean_cell_list(payload.get("useful_cells") or []) if judge != "no" else []
    return {
        "think": str(payload.get("think") or "").strip(),
        "useful_cells": useful_cells,
        "summary": str(payload.get("summary") or "").strip(),
        "judge": judge,
    }


def build_sft_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a visual-RAG agent. Each iteration: decide whether to search for one more page "
                "or to answer; if searching, output a short query; for every retrieved page emit an "
                "analyse JSON with think, useful_cells, summary, judge. "
                "Decide outputs must be JSON with think, action, content. "
                "For analyse judge=no, useful_cells must be an empty list."
            ),
        },
        {"role": "user", "content": record.get("question") or ""},
    ]
    for step in record.get("trace", []):
        kind = step.get("step")
        if kind == "decide":
            messages.append({"role": "assistant", "content": dump_json(clean_decide_payload(step.get("result") or {}))})
        elif kind == "search":
            messages.append(
                {
                    "role": "tool",
                    "name": "retrieve_pages",
                    "content": dump_json({"query": step.get("query"), "pages": step.get("pages", [])}),
                }
            )
        elif kind == "analyse":
            messages.append({"role": "assistant", "content": dump_json(clean_analyse_for_message(step.get("result") or {}))})
    return messages


def clean_analyse_for_message(payload: dict[str, Any]) -> dict[str, Any]:
    judge = payload.get("judge") if payload.get("judge") in {"yes", "partial", "no"} else "no"
    useful_cells = clean_cell_list(payload.get("useful_cells") or []) if judge != "no" else []
    return {
        "think": str(payload.get("think") or "").strip(),
        "useful_cells": useful_cells,
        "summary": str(payload.get("summary") or "").strip(),
        "judge": judge,
    }


def normalize_label_set(value: Any) -> set[str]:
    if isinstance(value, dict):
        value = value.values()
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item) for item in value if str(item)}


def page_labels_from_pages(pages: list[Any]) -> list[str]:
    labels: list[str] = []
    for page in pages:
        if isinstance(page, dict):
            label = page.get("page_label") or page.get("label") or page.get("id")
        else:
            label = page
        if label:
            labels.append(str(label))
    return labels


def first_retained_path(step: dict[str, Any]) -> str | None:
    retained = step.get("retained_images")
    if isinstance(retained, list) and retained:
        return str(retained[0])
    return None


def append_unique(target: list[str], values: list[str]) -> None:
    seen = set(target)
    for value in values or []:
        text = str(value).strip()
        if text and text not in seen:
            target.append(text)
            seen.add(text)


def clean_cell_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip().upper().replace(" ", "")
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def parse_json(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


if __name__ == "__main__":
    main()
