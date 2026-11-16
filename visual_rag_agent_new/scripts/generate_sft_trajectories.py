#!/usr/bin/env python
"""Teacher distillation for the new protocol (DESIGN.md §7): roll the Agent with a strong teacher,
keep the trajectories that pass the filter (answered on their own, no error turns, judge-correct,
gold pages subset of E_final = the pages attached to the question tree, DESIGN.md §11.8), and emit
one SFT row per applied turn with the exact inference context (messages / image_paths) and the
verbatim response as target. Each trajectory is tagged with what it exercised (revisit, fact
correction, expand, crop) so the trajectory types of §7 can be balanced afterwards.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.agent import Agent, is_fatal_model_error
from src.config import build_retriever, build_vlm, load_config
from src.dataset import (done_sample_ids, extract_answer, extract_question, extract_reference_pages, extract_sample_id,
                         page_key, read_jsonl, reference_labels, repair_jsonl_tail, retrieval_deck, safe_name)
from src.judge import judge_prediction_row, load_dotenv
from src.pipeline import rollout_then_judge
from src.retriever import LockedRetriever

ERROR_STEPS = {"format_error", "edit_error", "runner_error"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher trajectories and SFT rows for the new protocol.")
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--config", nargs="+", default=["config/teacher.yaml"], help="YAML files; later ones override earlier")
    parser.add_argument("--output-dir", default="outputs/sft")
    parser.add_argument("--run-dir", default=None,
                        help="write here instead of a new timestamped dir; samples already in its trajectories.jsonl are skipped")
    parser.add_argument("--index", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--retrieval-scope", choices=["flat", "deck"], default=None, help="default: agent.retrieval_scope")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--judge", choices=["none", "deepseek"], default="none")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--judge-max-tokens", type=int, default=None)
    parser.add_argument("--require-judge-correct", action="store_true")
    parser.add_argument("--require-all-reference-pages", action="store_true")
    parser.add_argument("--stop-on-api-error", action="store_true")
    parser.add_argument("--target-kept", type=int, default=0, help="stop submitting after this many kept trajectories (0 = all)")
    parser.add_argument("--workers", type=int, default=1, help="parallel rollouts (vLLM); the retriever is locked")
    parser.add_argument("--judge-workers", type=int, default=16, help="judge calls run on their own pool")
    args = parser.parse_args()

    load_dotenv()
    if args.require_judge_correct and args.judge == "none":
        raise ValueError("--require-judge-correct requires --judge deepseek")
    config = load_config(args.config)
    run_root = Path(args.run_dir) if args.run_dir else Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.dataset_file), start=args.start_index, limit=args.num_samples)
    items = list(enumerate(rows))
    if args.shuffle:
        random.Random(args.seed).shuffle(items)
    for name in ("trajectories", "kept_trajectories", "rejected_trajectories", "sft", "all_sft"):
        repair_jsonl_tail(run_root / f"{name}.jsonl")
    done = done_sample_ids(run_root / "trajectories.jsonl")
    items = [(i, r) for i, r in items if extract_sample_id(r, i + args.start_index) not in done]
    print(f"[sft-gen] run_dir={run_root} todo={len(items)} already_done={len(done)}", flush=True)

    retriever = build_retriever(config, index_path=args.index, device=args.device)
    if args.workers > 1:
        retriever = LockedRetriever(retriever)
    vlm = build_vlm(config, device=args.device)
    agent_cfg = config["agent"]
    scope = args.retrieval_scope or str(agent_cfg.get("retrieval_scope", "flat"))

    paths = {name: run_root / f"{name}.jsonl" for name in
             ("trajectories", "kept_trajectories", "rejected_trajectories", "sft", "all_sft")}
    stats: dict[str, Any] = {"retrieval_scope": scope, "skipped_done": len(done), "processed": 0, "kept": 0, "rejected": 0, "sft_rows": 0, "all_sft_rows": 0,
                             "reject_reasons": Counter(), "stopped_on_api_error": False}
    lock = threading.Lock()
    stop = threading.Event()

    def run_one(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        local_idx, row = item
        sample_id = extract_sample_id(row, local_idx + args.start_index)
        agent = Agent(vlm=vlm, retriever=retriever, max_turns=int(agent_cfg["max_turns"]),
                      retrieve_k=int(agent_cfg["retrieve_k"]), frame=str(agent_cfg["frame"]),
                      sft_frame=agent_cfg.get("sft_frame"), first_search=str(agent_cfg["first_search"]),
                      search_guidance=bool(agent_cfg["search_guidance"]),
                      teacher_hint=str(agent_cfg.get("teacher_hint") or ""),
                      image_max_pixels=agent_cfg.get("image_max_pixels"))
        try:
            result = agent.run(extract_question(row), output_dir=run_root / "samples" / safe_name(sample_id),
                               deck_name=retrieval_deck(row, scope))
        except Exception as exc:  # noqa: BLE001
            if is_fatal_model_error(exc):
                raise
            result = {"query": extract_question(row), "answer": "", "final_pages": [], "memory": {},
                      "counts": {}, "turns": 0, "terminated_by": "runner_error",
                      "trace": [{"step": "runner_error", "error": str(exc)}]}
        result["sample_id"] = sample_id
        result["source_row_index"] = local_idx + args.start_index
        result["gold_answer"] = extract_answer(row)
        result["hop"] = row.get("hop")
        result["evidence_pages"] = sorted(extract_reference_pages(row))
        result["reference_pages"] = reference_labels(row)
        result["tags"] = trajectory_tags(result)
        return result

    def judge(result: dict[str, Any]) -> dict[str, Any]:
        if args.judge == "deepseek" and result.get("answer"):
            try:
                result["judge"] = judge_prediction_row(result, model=args.judge_model, base_url=args.judge_base_url,
                                                       timeout=args.judge_timeout, max_retries=args.judge_max_retries,
                                                       max_tokens=args.judge_max_tokens)
            except Exception as exc:  # noqa: BLE001
                result["judge_error"] = str(exc)
        return result

    def record(result: dict[str, Any]) -> None:
        with lock:
            stats["processed"] += 1
            if args.stop_on_api_error and has_api_error(result):
                result["sft_keep"], result["sft_reject_reasons"] = False, ["api_error"]
                append_jsonl(paths["trajectories"], result)
                append_jsonl(paths["rejected_trajectories"], result)
                stats["rejected"] += 1
                stats["reject_reasons"]["api_error"] += 1
                stats["stopped_on_api_error"] = True
                stop.set()
                print(f"[sft-gen] stopping on API error: {first_trace_error(result)}", flush=True)
                return
            keep, reasons = should_keep_trajectory(result, require_judge_correct=args.require_judge_correct,
                                                   require_all_reference_pages=args.require_all_reference_pages)
            result["sft_keep"], result["sft_reject_reasons"] = keep, reasons
            calls = build_sft_calls(result)
            for call in calls:
                append_jsonl(paths["all_sft"], call)
                if keep:
                    append_jsonl(paths["sft"], call)
            stats["all_sft_rows"] += len(calls)
            if keep:
                stats["kept"] += 1
                stats["sft_rows"] += len(calls)
                append_jsonl(paths["kept_trajectories"], result)
                if args.target_kept and stats["kept"] >= args.target_kept:
                    stop.set()
            else:
                stats["rejected"] += 1
                stats["reject_reasons"].update(r.split(":", 1)[0] for r in reasons)
                append_jsonl(paths["rejected_trajectories"], result)
            append_jsonl(paths["trajectories"], result)
            print(f"[sft-gen] {stats['processed']}/{len(items)} sample_id={result.get('sample_id')} "
                  f"terminated_by={result.get('terminated_by')} keep={keep} reasons={reasons} sft={len(calls)}", flush=True)

    rollout_then_judge(items, run_one, judge, record, workers=args.workers, judge_workers=args.judge_workers, stop=stop)

    stats["reject_reasons"] = dict(stats["reject_reasons"])
    stats["totals"] = run_totals(read_jsonl(paths["trajectories"]) if paths["trajectories"].exists() else [])
    (run_root / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_root": str(run_root), "stats": stats}, ensure_ascii=False, indent=2))


# ------------------------------------------------------------------ extraction / filter


def run_totals(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    """Cumulative keep statistics over every trajectory of a (possibly resumed) run."""
    kept = [t for t in trajectories if t.get("sft_keep")]
    reasons: Counter = Counter()
    for t in trajectories:
        if not t.get("sft_keep"):
            reasons.update(r.split(":", 1)[0] for r in t.get("sft_reject_reasons") or [])
    tag_names = sorted({k for t in kept for k in (t.get("tags") or {})})
    return {
        "processed": len(trajectories),
        "kept": len(kept),
        "processed_by_hop": dict(Counter(str(t.get("hop")) for t in trajectories)),
        "kept_by_hop": dict(Counter(str(t.get("hop")) for t in kept)),
        "reject_reasons": dict(reasons),
        "kept_with": {k: sum(1 for t in kept if (t.get("tags") or {}).get(k)) for k in tag_names},
    }


def build_sft_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    """One SFT row per applied turn; each carries the exact inference context from the trace
    (no prompt reconstruction). The context only holds memory committed before that turn."""
    sample_id = str(result.get("sample_id", ""))
    calls: list[dict[str, Any]] = []
    for step in result.get("trace", []):
        if step.get("step") != "turn":
            continue
        sft = step.get("sft") or {}
        target = str(sft.get("target") or "").strip()
        messages = sft.get("messages") or []
        if not target or not messages:
            continue
        call = {
            "id": f"{sample_id}:turn:{step.get('turn')}",
            "sample_id": sample_id,
            "turn": step.get("turn"),
            "view": step.get("view"),
            "action": (step.get("act") or {}).get("action", {}).get("type"),
            "messages": messages,
            "image_paths": [str(p) for p in sft.get("image_paths") or []],
            "target": target,
        }
        call["conversations"] = to_qwenvl_conversations(messages, target)
        calls.append(call)
    return calls


def should_keep_trajectory(result: dict[str, Any], *, require_judge_correct: bool, require_all_reference_pages: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if str(result.get("terminated_by") or "") != "answer":
        reasons.append(f"terminated_by:{result.get('terminated_by')}")
    errors = sorted({str(s.get("step")) for s in result.get("trace") or [] if str(s.get("step")) in ERROR_STEPS})
    if errors:
        reasons.append("error_steps:" + ",".join(errors))
    if not str(result.get("answer") or "").strip():
        reasons.append("empty_answer")
    if require_judge_correct:
        judge = result.get("judge")
        if not isinstance(judge, dict):
            reasons.append("judge_missing")
        elif judge.get("correct") is not True:
            reasons.append("judge_incorrect")
    if require_all_reference_pages:
        gold = result.get("reference_pages") or []
        final = {page_key(label) for label in result.get("final_pages") or []}
        missing = [label for label in gold if page_key(label) not in final]
        if not gold:
            reasons.append("reference_pages_missing")
        elif missing:
            reasons.append("missing_reference_pages:" + ",".join(missing))
    return not reasons, reasons


def trajectory_tags(result: dict[str, Any]) -> dict[str, int]:
    tags = {"revisit": 0, "crop": 0, "update_fact": 0, "expand": 0}
    for step in result.get("trace") or []:
        if step.get("step") != "turn":
            continue
        action = (step.get("act") or {}).get("action", {}).get("type")
        if action in ("revisit", "crop"):
            tags[action] += 1
        events = step.get("events") or {}
        tags["update_fact"] += len(events.get("updated_facts") or [])
        tags["expand"] += 1 if events.get("expanded") else 0
    return tags


def to_qwenvl_conversations(messages: list[dict[str, str]], target: str) -> list[dict[str, str]]:
    role_map = {"system": "system", "user": "human", "assistant": "gpt"}
    out = [{"from": role_map.get(str(m.get("role")), "human"), "value": str(m.get("content") or "")} for m in messages]
    out.append({"from": "gpt", "value": target})
    return out


def has_api_error(result: dict[str, Any]) -> bool:
    text = first_trace_error(result).lower()
    markers = ("api ", "quota", "balance", "rate limit", "rate_limit", "too many requests", "billing",
               "invalid api key", "model not found", "no permission", "access denied", "forbidden", "http 5")
    return any(m in text for m in markers)


def first_trace_error(result: dict[str, Any]) -> str:
    for step in result.get("trace") or []:
        if step.get("error"):
            return str(step["error"])
    return str(result.get("judge_error") or "")


# ------------------------------------------------------------------ io helpers


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
