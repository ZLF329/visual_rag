#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from src.image_utils import crop_grid_cells_bbox, resize_to_pixel_count
from src.prompt_serialization import build_agent_human_turn
from src.prompts import build_analyse_prompt, build_decide_prompt, build_evidence_update_prompt


DEFAULT_SOURCE = (
    "outputs/sft_trajectories/"
    "qwen36plus_new3step_900single300multi_stepfiltered_20260529/"
    "kept_balanced_1200.jsonl"
)
DEFAULT_WARMUP = (
    "outputs/sft_qwenvl/"
    "qwen36plus_new3step_relevance_head_warmup200.jsonl"
)
DEFAULT_JOINT = (
    "outputs/sft_qwenvl/"
    "qwen36plus_new3step_relevance_joint1000.jsonl"
)
DEFAULT_CROP_ROOT = "outputs/sft_qwenvl/relevance_bbox_crops"

CELL_RE = re.compile(r"^([A-H])([1-8])$")
COLS = "ABCDEFGH"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SlideVQA agent trajectories into the no-grid analyse format "
            "with 8x8 relevance labels for QwenVL SFT."
        )
    )
    parser.add_argument("--source-file", default=DEFAULT_SOURCE)
    parser.add_argument("--warmup-output", default=DEFAULT_WARMUP)
    parser.add_argument("--joint-output", default=DEFAULT_JOINT)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--crop-root", default=DEFAULT_CROP_ROOT)
    parser.add_argument("--retained-crop-pixels", type=int, default=400_000)
    parser.add_argument("--warmup-trajectories", type=int, default=200)
    parser.add_argument("--joint-trajectories", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=True)
    parser.add_argument("--joint-cell-loss-weight", type=float, default=0.3)
    args = parser.parse_args()

    source_path = Path(args.source_file)
    trajectories = [json.loads(line) for line in source_path.open(encoding="utf-8")]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(trajectories)

    needed = args.warmup_trajectories + args.joint_trajectories
    if len(trajectories) < needed:
        raise ValueError(f"need {needed} trajectories, found {len(trajectories)}")

    warmup_trajs = trajectories[: args.warmup_trajectories]
    joint_trajs = trajectories[
        args.warmup_trajectories : args.warmup_trajectories + args.joint_trajectories
    ]

    warmup_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "source_file": str(source_path),
        "shuffle": args.shuffle,
        "seed": args.seed if args.shuffle else None,
        "warmup_trajectories": len(warmup_trajs),
        "joint_trajectories": len(joint_trajs),
        "warmup_hop_counts": Counter(),
        "joint_hop_counts": Counter(),
        "warmup_step_counts": Counter(),
        "joint_step_counts": Counter(),
        "judge_counts": Counter(),
        "invalid_cells": Counter(),
    }

    for split_name, trajs in (("warmup", warmup_trajs), ("joint", joint_trajs)):
        for traj_idx, traj in enumerate(trajs):
            hop = traj.get("hop_type") or traj.get("source_hop_type") or "unknown"
            stats[f"{split_name}_hop_counts"][hop] += 1
            retained_crop_paths: list[str] = []
            successful_history: list[str] = []
            failed_history: list[str] = []
            evidence_state: dict[str, Any] = empty_evidence_state()
            trace_steps = list(traj.get("trace") or [])
            last_non_no_analyse_idx = None
            if str(hop).lower() == "multi":
                for idx, trace_step in enumerate(trace_steps):
                    if trace_step.get("step") != "analyse":
                        continue
                    trace_result = dict(trace_step.get("result") or {})
                    trace_judge = trace_result.get("judge") or trace_result.get("decision") or "no"
                    if trace_judge != "no":
                        last_non_no_analyse_idx = idx

            for step_idx, step in enumerate(trace_steps):
                step_name = step.get("step")
                if split_name == "warmup":
                    if step_name != "analyse":
                        continue
                    row = build_analyse_row(
                        traj,
                        step,
                        row_id=f"warmup:{traj.get('sample_id')}:{step_idx}",
                        lm_loss_weight=0.0,
                        cell_loss_weight=1.0,
                        invalid_cells=stats["invalid_cells"],
                    )
                    warmup_rows.append(row)
                    stats["warmup_step_counts"]["analyse"] += 1
                    stats["judge_counts"][row.get("judge", "unknown")] += 1
                    continue

                if step_name == "decide":
                    row = build_decide_row(
                        traj,
                        step,
                        row_id=f"joint:{traj.get('sample_id')}:{step_idx}",
                        retained_image_paths=retained_crop_paths,
                        memory_context=format_memory_context(
                            successful_history,
                            failed_history,
                            evidence_state,
                        ),
                    )
                    joint_rows.append(row)
                    stats["joint_step_counts"]["decide"] += 1
                elif step_name == "search":
                    maybe_append_failed_search(step, failed_history)
                elif step_name == "analyse":
                    row = build_analyse_row(
                        traj,
                        step,
                        row_id=f"joint:{traj.get('sample_id')}:{step_idx}",
                        lm_loss_weight=1.0,
                        cell_loss_weight=args.joint_cell_loss_weight,
                        invalid_cells=stats["invalid_cells"],
                    )
                    joint_rows.append(row)
                    stats["joint_step_counts"]["analyse"] += 1
                    stats["judge_counts"][row.get("judge", "unknown")] += 1
                    crop_path = maybe_materialize_retained_crop(
                        row=row,
                        traj=traj,
                        step_idx=step_idx,
                        crop_root=Path(args.crop_root),
                        retained_crop_pixels=args.retained_crop_pixels,
                        stats=stats,
                    )
                    previous_state = json_roundtrip(public_evidence_state(evidence_state))
                    source_judge = str(row.get("judge") or "no")
                    target_judge = source_judge
                    if step_idx == last_non_no_analyse_idx:
                        target_judge = "yes"
                    target_state = synthesize_evidence_state_after_analyse(
                        evidence_state,
                        step,
                        target_judge=target_judge,
                    )
                    update_row = build_evidence_update_row(
                        traj,
                        step,
                        row_id=f"joint:{traj.get('sample_id')}:{step_idx}:evidence_update",
                        retained_image_path=crop_path or (row.get("image") or [""])[0],
                        previous_evidence_state=previous_state,
                        target_evidence_state=target_state,
                        target_judge=target_judge,
                    )
                    joint_rows.append(update_row)
                    stats["joint_step_counts"]["evidence_update"] += 1
                    stats.setdefault("evidence_update_judge_counts", Counter())
                    stats["evidence_update_judge_counts"][target_judge] += 1
                    if target_judge == "no":
                        update_histories_after_analyse(
                            step,
                            successful_history,
                            failed_history,
                            evidence_state,
                            judge_override=target_judge,
                        )
                        continue

                    evidence_state.clear()
                    evidence_state.update(json_roundtrip(target_state))
                    update_histories_after_analyse(
                        step,
                        successful_history,
                        failed_history,
                        evidence_state,
                        judge_override=target_judge,
                    )
                    if crop_path is not None:
                        retained_crop_paths.append(crop_path)
                elif step_name == "evidence_update":
                    continue

    write_jsonl(Path(args.warmup_output), warmup_rows)
    write_jsonl(Path(args.joint_output), joint_rows)

    stats["warmup_rows"] = len(warmup_rows)
    stats["joint_rows"] = len(joint_rows)
    stats["crop_root"] = args.crop_root
    stats["warmup_output"] = args.warmup_output
    stats["joint_output"] = args.joint_output
    stats = json_roundtrip(stats)

    summary_path = Path(args.summary_output) if args.summary_output else Path(args.joint_output).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def build_decide_row(
    traj: dict[str, Any],
    step: dict[str, Any],
    *,
    row_id: str,
    retained_image_paths: list[str] | None = None,
    memory_context: str | None = None,
) -> dict[str, Any]:
    retained_image_paths = list(retained_image_paths or [])
    system, user = build_decide_prompt(
        original_query=str(traj.get("question") or ""),
        memory_context=str(memory_context if memory_context is not None else step.get("memory_context") or ""),
        retained_visual_evidence_count=len(retained_image_paths),
    )
    target = sanitize_decide_target(step.get("result") or {})
    return {
        "id": row_id,
        "sample_id": traj.get("sample_id"),
        "source_hop_type": traj.get("hop_type") or traj.get("source_hop_type"),
        "step": "decide",
        "image": retained_image_paths,
        "cell_labels": [0.0] * 64,
        "cell_loss_mask": 0.0,
        "lm_loss_weight": 1.0,
        "cell_loss_weight": 0.0,
        "conversations": [
            {
                "from": "human",
                "value": build_agent_human_turn(system, user, call_type="decide"),
            },
            {"from": "gpt", "value": json.dumps(target, ensure_ascii=False)},
        ],
    }


def build_analyse_row(
    traj: dict[str, Any],
    step: dict[str, Any],
    *,
    row_id: str,
    lm_loss_weight: float,
    cell_loss_weight: float,
    invalid_cells: Counter,
) -> dict[str, Any]:
    _, user = build_analyse_prompt(
        original_query=str(traj.get("question") or ""),
    )
    result = dict(step.get("result") or {})
    judge = result.get("judge") or result.get("decision") or "no"
    cells = [] if judge == "no" else list(result.get("useful_cells") or [])
    labels = cells_to_labels(cells, invalid_cells=invalid_cells)
    target = {
        "think": str(result.get("think") or ""),
        "summary": str(result.get("summary") or ""),
    }
    image_path = step.get("source_image_path") or step.get("image_path")
    if not image_path:
        pages = [p for p in step.get("pages") or [] if isinstance(p, dict)]
        image_path = pages[0].get("image_path") if pages else None
    if not image_path:
        raise ValueError(f"analyse step has no source image: {row_id}")
    return {
        "id": row_id,
        "sample_id": traj.get("sample_id"),
        "source_hop_type": traj.get("hop_type") or traj.get("source_hop_type"),
        "step": "analyse",
        "judge": judge,
        "search_query": step.get("query"),
        "image": [str(Path(image_path).resolve())],
        "source_image_path": str(Path(image_path).resolve()),
        "cell_labels": labels,
        "useful_cells": cells,
        "cell_loss_mask": 1.0,
        "lm_loss_weight": float(lm_loss_weight),
        "cell_loss_weight": float(cell_loss_weight),
        "conversations": [
            {"from": "human", "value": user},
            {"from": "gpt", "value": json.dumps(target, ensure_ascii=False)},
        ],
    }


def build_evidence_update_row(
    traj: dict[str, Any],
    step: dict[str, Any],
    *,
    row_id: str,
    retained_image_path: str,
    previous_evidence_state: dict[str, Any],
    target_evidence_state: dict[str, Any],
    target_judge: str,
) -> dict[str, Any]:
    result = dict(step.get("result") or {})
    system, user = build_evidence_update_prompt(
        original_question=str(traj.get("question") or ""),
        search_query=str(step.get("query") or ""),
        page_summary=observed_evidence_from_result(result) or str(result.get("summary") or ""),
        evidence_state_json=json.dumps(previous_evidence_state, ensure_ascii=False, indent=2),
    )
    target = {
        "evidence_state": public_evidence_state(target_evidence_state),
        "judge": str(target_judge if target_judge in {"yes", "partial", "no"} else "partial"),
    }
    return {
        "id": row_id,
        "sample_id": traj.get("sample_id"),
        "source_hop_type": traj.get("hop_type") or traj.get("source_hop_type"),
        "step": "evidence_update",
        "image": [str(Path(retained_image_path).resolve())],
        "cell_labels": [0.0] * 64,
        "cell_loss_mask": 0.0,
        "lm_loss_weight": 1.0,
        "cell_loss_weight": 0.0,
        "conversations": [
            {
                "from": "human",
                "value": build_agent_human_turn(system, user, call_type="evidence_update"),
            },
            {"from": "gpt", "value": json.dumps(target, ensure_ascii=False)},
        ],
    }


def maybe_materialize_retained_crop(
    *,
    row: dict[str, Any],
    traj: dict[str, Any],
    step_idx: int,
    crop_root: Path,
    retained_crop_pixels: int,
    stats: dict[str, Any],
) -> str | None:
    if row.get("judge") == "no":
        return None
    image_paths = row.get("image") or []
    if not image_paths:
        return None
    source_path = Path(str(image_paths[0]))
    cells = [str(cell) for cell in row.get("useful_cells") or []]
    sample_id = str(traj.get("sample_id") or "unknown").replace("/", "_")
    crop_root.mkdir(parents=True, exist_ok=True)
    out_path = crop_root / f"{sample_id}_step{step_idx:03d}_retained.png"

    with Image.open(source_path) as image:
        crop = crop_grid_cells_bbox(image, cells)
        crop = resize_to_pixel_count(crop, retained_crop_pixels, allow_upscale=True)
        crop.save(out_path)

    stats.setdefault("retained_crop_count", 0)
    stats["retained_crop_count"] += 1
    if not cells:
        stats.setdefault("retained_full_page_fallback_count", 0)
        stats["retained_full_page_fallback_count"] += 1
    return str(out_path.resolve())


def observed_evidence_from_result(result: dict[str, Any]) -> str:
    text = str(result.get("observed_evidence") or "").strip()
    if text:
        return text
    return str(result.get("summary") or "").strip()


def empty_evidence_state() -> dict[str, Any]:
    return {
        "answer_relevant_facts": [],
        "missing_requirements": ["No evidence has been gathered yet."],
    }


def public_evidence_state(evidence_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer_relevant_facts": compact_unique_strings(
            list(evidence_state.get("answer_relevant_facts") or [])
        ),
        "missing_requirements": compact_unique_strings(
            list(evidence_state.get("missing_requirements") or [])
        ),
    }


def synthesize_evidence_state_after_analyse(
    evidence_state: dict[str, Any],
    step: dict[str, Any],
    *,
    target_judge: str,
) -> dict[str, Any]:
    state = public_evidence_state(evidence_state)
    if target_judge == "no":
        return state

    result = dict(step.get("result") or {})
    fact = observed_evidence_from_result(result) or str(result.get("summary") or "").strip()
    if fact and fact.lower() not in {item.lower() for item in state["answer_relevant_facts"]}:
        state["answer_relevant_facts"].append(fact)
    if target_judge == "yes":
        state["missing_requirements"] = []
    elif not state["missing_requirements"]:
        state["missing_requirements"] = [
            "Additional answer-relevant evidence is needed to fully answer the question."
        ]
    return public_evidence_state(state)


def compact_unique_strings(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").strip().split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(text)
    return rows


def format_memory_context(
    successful_history: list[str],
    failed_history: list[str],
    evidence_state: dict[str, Any],
) -> str:
    return (
        "Successful search history (yes/partial):\n"
        f"{format_history(successful_history)}\n\n"
        "Failed search history (do not repeat these queries):\n"
        f"{format_history(failed_history)}\n\n"
        "Current evidence_state:\n"
        f"{json.dumps(public_evidence_state(evidence_state), ensure_ascii=False, indent=2)}"
    )


def format_history(rows: list[str]) -> str:
    return "\n".join(rows[-8:]) if rows else "None yet."


def format_query_summary(query: str | None, summary: str) -> str:
    query_text = str(query or "").strip() or "unknown query"
    summary_text = str(summary or "").strip()
    if not summary_text:
        summary_text = "This search did not add usable evidence."
    return f"[query: {query_text}]: {summary_text}"


def maybe_append_failed_search(step: dict[str, Any], failed_history: list[str]) -> None:
    query = str(step.get("query") or "").strip()
    pages = step.get("pages")
    if pages is None:
        return
    if not pages:
        failed_history.append(format_query_summary(query, "No pages were retrieved; try a different query."))
        return
    new_pages = step.get("new_pages")
    if new_pages == []:
        failed_history.append(format_query_summary(query, "Only repeated pages found; try a different query."))


def update_histories_after_analyse(
    step: dict[str, Any],
    successful_history: list[str],
    failed_history: list[str],
    evidence_state: dict[str, Any],
    *,
    judge_override: str | None = None,
) -> None:
    result = dict(step.get("result") or {})
    judge = judge_override or result.get("judge") or result.get("decision") or "no"
    query = str(step.get("query") or "").strip()
    summary = str(result.get("summary") or "").strip()
    if judge == "no":
        failed_history.append(format_query_summary(query, summary))
        return

    successful_history.append(format_query_summary(query, summary))


def update_evidence_state_from_step(step: dict[str, Any], evidence_state: dict[str, Any]) -> None:
    result = dict(step.get("result") or {})
    state = result.get("evidence_state")
    if not isinstance(state, dict):
        return
    evidence_state.clear()
    evidence_state.update(public_evidence_state(state))


def cells_to_labels(cells: list[str], *, invalid_cells: Counter) -> list[float]:
    labels = [0.0] * 64
    for cell in cells:
        text = str(cell).strip().upper()
        match = CELL_RE.match(text)
        if not match:
            invalid_cells[text] += 1
            continue
        col, row = match.groups()
        idx = (int(row) - 1) * 8 + COLS.index(col)
        labels[idx] = 1.0
    return labels


def sanitize_decide_target(result: dict[str, Any]) -> dict[str, Any]:
    target = {
        "think": clean_decide_think(result.get("think") or ""),
        "action": result.get("action") or "search",
        "content": result.get("content") or result.get("search_query") or "",
    }
    return target


def clean_decide_think(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    patterns = [
        re.compile(r"\b(?:there is )?no remaining gaps?\b\.?\s*", flags=re.IGNORECASE),
        re.compile(r"\b(?:the )?remaining gaps?\b[^.]*\.?\s*", flags=re.IGNORECASE),
        re.compile(r"\b(?:the )?remaining_gaps?\b[^.]*\.?\s*", flags=re.IGNORECASE),
    ]
    for pattern in patterns:
        text = pattern.sub("", text)
    return " ".join(text.split()).strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
