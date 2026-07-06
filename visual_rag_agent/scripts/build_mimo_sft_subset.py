#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.prompts import build_analyse_prompt, build_decide_prompt, build_evidence_update_prompt
from src.schemas import EvidenceState


DEFAULT_RUN_ROOT = (
    "outputs/sft_trajectories/mimo25_train_balanced_1200_calllevel_20260527_172453"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/sft_trajectories/mimo25_train_300multi_shortest_singlefill_calllevel"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a MiMo SFT subset with shortest multi-hop trajectories and single-hop fill."
    )
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-total", type=int, default=1200)
    parser.add_argument("--multi-count", type=int, default=300)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    multi_records = unique_records(load_kept_records(run_root / "multi"))
    single_records = unique_records(
        record
        for subdir in sorted(run_root.glob("single*"))
        if subdir.is_dir()
        for record in load_kept_records(subdir)
    )

    selected_multi = sorted(multi_records.values(), key=sort_key)[: args.multi_count]
    needed_single = max(0, args.target_total - len(selected_multi))
    selected_single = sorted(single_records.values(), key=sort_key)[:needed_single]
    selected = selected_single + selected_multi

    trajectories_path = output_dir / "kept_selected_trajectories.jsonl"
    calls_path = output_dir / "kept_selected_sft_calls.jsonl"
    summary_path = output_dir / "summary.json"

    calls = records_to_sft_calls(selected)
    write_jsonl(trajectories_path, selected)
    write_jsonl(calls_path, calls)

    summary = build_summary(
        run_root=run_root,
        output_dir=output_dir,
        multi_records=multi_records,
        single_records=single_records,
        selected_multi=selected_multi,
        selected_single=selected_single,
        calls=calls,
        target_total=args.target_total,
        multi_count=args.multi_count,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_kept_records(directory: Path) -> list[dict[str, Any]]:
    raw_path = directory / "raw_trajectories.jsonl"
    kept_path = directory / "kept_sft_trajectories.jsonl"
    path = raw_path if raw_path.exists() else kept_path
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for row in read_jsonl(path):
        if row.get("keep", True):
            records.append(row)
    return records


def unique_records(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record.get("sample_id") or record.get("id") or record.get("row_index"))
        current = out.get(sample_id)
        if current is None or sort_key(record) < sort_key(current):
            out[sample_id] = record
    return out


def sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    return (
        call_count(record),
        int(record.get("retrieval_steps") or analyse_count(record) or 0),
        str(record.get("sample_id") or record.get("id") or record.get("row_index")),
    )


def call_count(record: dict[str, Any]) -> int:
    return sum(1 for step in record.get("trace", []) if step.get("step") in {"decide", "analyse"})


def analyse_count(record: dict[str, Any]) -> int:
    return sum(1 for step in record.get("trace", []) if step.get("step") == "analyse")


def records_to_sft_calls(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for record in records:
        calls.extend(build_sft_calls(record))
    return calls


def build_sft_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    question = str(record.get("question") or "")
    sample_id = str(record.get("sample_id") or record.get("row_index") or "unknown")
    for call_idx, step in enumerate(record.get("trace", [])):
        kind = step.get("step")
        if kind == "decide":
            memory_context = str(step.get("memory_context") or default_memory_context())
            system, user = build_decide_prompt(
                original_query=question,
                memory_context=memory_context,
            )
            target = step.get("raw_text") or json.dumps(step.get("result") or {}, ensure_ascii=False)
            image_paths = [str(path) for path in step.get("retained_images") or []]
        elif kind == "analyse":
            grid_image_path = str(step.get("grid_image_path") or "")
            if not grid_image_path:
                continue
            system, user = build_analyse_prompt(
                original_query=question,
                search_query=str(step.get("query") or ""),
            )
            target = step.get("raw_text") or json.dumps(step.get("result") or {}, ensure_ascii=False)
            image_paths = [grid_image_path]
        else:
            continue

        calls.append(
            {
                "id": f"{sample_id}:{kind}:{call_idx}",
                "sample_id": sample_id,
                "row_index": record.get("row_index"),
                "step": kind,
                "iter": step.get("iter"),
                "source_keep": bool(record.get("keep", True)),
                "source_hop_type": hop_type(record),
                "page": step.get("page"),
                "query": step.get("query"),
                "system": system,
                "user": user,
                "image_paths": image_paths,
                "target": str(target).strip(),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": str(target).strip()},
                ],
            }
        )
    return calls


def default_memory_context() -> str:
    return (
        "Search history:\n"
        "None yet.\n\n"
        "Current evidence_state:\n"
        + EvidenceState().model_dump_json(indent=2)
    )


def hop_type(record: dict[str, Any]) -> str:
    explicit = str(record.get("hop_type") or "").lower()
    if "multi" in explicit:
        return "multi"
    if "single" in explicit:
        return "single"
    refs = record.get("reference_page_labels") or record.get("reference_pages") or []
    return "multi" if len(refs) > 1 else "single"


def build_summary(
    *,
    run_root: Path,
    output_dir: Path,
    multi_records: dict[str, dict[str, Any]],
    single_records: dict[str, dict[str, Any]],
    selected_multi: list[dict[str, Any]],
    selected_single: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    target_total: int,
    multi_count: int,
) -> dict[str, Any]:
    call_hops = Counter(call.get("source_hop_type") for call in calls)
    call_steps = Counter(call.get("step") for call in calls)
    trajectory_calls = Counter(call_count(record) for record in selected_single + selected_multi)
    return {
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "target_total": target_total,
        "target_multi": multi_count,
        "available_single": len(single_records),
        "available_multi": len(multi_records),
        "selected_total": len(selected_single) + len(selected_multi),
        "selected_single": len(selected_single),
        "selected_multi": len(selected_multi),
        "single_shortfall": max(0, target_total - multi_count - len(selected_single)),
        "trajectory_call_count_distribution": dict(sorted(trajectory_calls.items())),
        "sft_calls_total": len(calls),
        "sft_calls_by_hop": dict(call_hops),
        "sft_calls_by_step": dict(call_steps),
        "files": {
            "trajectories": str(output_dir / "kept_selected_trajectories.jsonl"),
            "sft_calls": str(output_dir / "kept_selected_sft_calls.jsonl"),
            "summary": str(output_dir / "summary.json"),
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
