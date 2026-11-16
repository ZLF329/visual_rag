#!/usr/bin/env python
"""Run the new agent over a JSONL dataset -> <output>/<timestamp>/predictions.jsonl + summary.json.
Each prediction row is the agent's run output plus sample_id / gold_answer (/ judge), which is what
eval/judge_batch.py and the scoring scripts consume.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.agent import Agent, is_fatal_model_error
from src.config import build_retriever, build_vlm, load_config
from src.dataset import (done_sample_ids, extract_answer, extract_question, extract_reference_pages, extract_sample_id,
                         page_key, read_jsonl, reference_labels, repair_jsonl_tail, retrieval_deck, safe_name)
from src.judge import add_judge_summary, judge_prediction_row, load_dotenv
from src.pipeline import rollout_then_judge
from src.retriever import LockedRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the new agent on a JSONL dataset.")
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--config", nargs="+", default=["config/default.yaml"], help="YAML files; later ones override earlier")
    parser.add_argument("--output", default="outputs/runs")
    parser.add_argument("--run-dir", default=None,
                        help="write here instead of a new timestamped dir; samples already in its predictions.jsonl are skipped")
    parser.add_argument("--index", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--retrieval-scope", choices=["flat", "deck"], default=None, help="default: agent.retrieval_scope")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--judge", choices=["none", "deepseek"], default="none")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--judge-max-tokens", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1, help="parallel rollouts (vLLM); the retriever is locked")
    parser.add_argument("--judge-workers", type=int, default=16, help="judge calls run on their own pool")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    run_root = Path(args.run_dir) if args.run_dir else Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    predictions_path = run_root / "predictions.jsonl"
    examples = read_jsonl(Path(args.dataset_file), start=args.start_index, limit=args.num_samples)
    repair_jsonl_tail(predictions_path)
    done = done_sample_ids(predictions_path)
    pairs = [(i, r) for i, r in enumerate(examples) if extract_sample_id(r, i + args.start_index) not in done]
    print(f"[eval] run_dir={run_root} todo={len(pairs)} already_done={len(done)}", flush=True)

    retriever = build_retriever(config, index_path=args.index, device=args.device)
    if args.concurrency > 1:
        retriever = LockedRetriever(retriever)
    vlm = build_vlm(config, device=args.device)
    agent_cfg = config["agent"]
    scope = args.retrieval_scope or str(agent_cfg.get("retrieval_scope", "flat"))

    rows: list[dict[str, Any]] = []
    lock = threading.Lock()

    def run_one(pair: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, row = pair
        sample_id = extract_sample_id(row, idx + args.start_index)
        agent = Agent(
            vlm=vlm, retriever=retriever,
            max_turns=int(agent_cfg["max_turns"]), retrieve_k=int(agent_cfg["retrieve_k"]),
            frame=str(agent_cfg["frame"]), first_search=str(agent_cfg["first_search"]),
            search_guidance=bool(agent_cfg["search_guidance"]), image_max_pixels=agent_cfg.get("image_max_pixels"),
            format_retries=int(agent_cfg.get("format_retries", 0)),
        )
        try:
            result = agent.run(extract_question(row), output_dir=run_root / "samples" / safe_name(sample_id),
                               deck_name=retrieval_deck(row, scope))
        except Exception as exc:  # noqa: BLE001
            if is_fatal_model_error(exc):
                raise
            result = {"query": extract_question(row), "answer": "", "final_pages": [], "memory": {},
                      "counts": {}, "turns": 0, "trace": [{"step": "runner_error", "error": str(exc)}],
                      "terminated_by": "runner_error"}
        result["sample_id"] = sample_id
        result["hop"] = row.get("hop")
        result["gold_answer"] = extract_answer(row)
        result["evidence_pages"] = sorted(extract_reference_pages(row))
        result["reference_pages"] = reference_labels(row)
        return result

    def judge(result: dict[str, Any]) -> dict[str, Any]:
        if args.judge == "deepseek":
            try:
                result["judge"] = judge_prediction_row(result, model=args.judge_model, base_url=args.judge_base_url,
                                                       timeout=args.judge_timeout, max_retries=args.judge_max_retries,
                                                       max_tokens=args.judge_max_tokens)
            except Exception as exc:  # noqa: BLE001
                result["judge_error"] = str(exc)
        return result

    def finish(result: dict[str, Any]) -> None:
        with lock:
            rows.append(result)
            with predictions_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            print(f"[eval] {len(rows)}/{len(pairs)} sample_id={result['sample_id']} terminated_by={result.get('terminated_by')} "
                  f"counts={result.get('counts')}", flush=True)

    rollout_then_judge(pairs, run_one, judge, finish, workers=args.concurrency, judge_workers=args.judge_workers)

    all_rows = read_jsonl(predictions_path)            # this session's rows plus any resumed ones
    summary = summarize(all_rows)
    summary["retrieval_scope"] = scope
    add_judge_summary(summary, all_rows)
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_root": str(run_root), "summary": summary}, ensure_ascii=False, indent=2))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Behaviour stats next to accuracy (SCORE Table 3/4 style): visits, final-evidence coverage,
    non-gold final pages, budget exhaustion."""
    n = max(1, len(rows))
    terminated = Counter(str(r.get("terminated_by")) for r in rows)
    counts = [r.get("counts") or {} for r in rows]
    coverage, noise = [], []
    for r in rows:
        gold = {page_key(p) for p in r.get("reference_pages") or []}
        final = {page_key(p) for p in r.get("final_pages") or []}
        if gold:
            coverage.append(1.0 if gold.issubset(final) else 0.0)
            noise.append(len(final - gold))
    return {
        "num_samples": len(rows),
        "terminated_by": dict(terminated),
        "avg_turns": sum(r.get("turns", 0) for r in rows) / n,
        "avg_searches": sum(c.get("search", 0) for c in counts) / n,
        "avg_revisits": sum(c.get("revisit", 0) for c in counts) / n,
        "avg_crops": sum(c.get("crop", 0) for c in counts) / n,
        "revisit_rate": sum(1 for c in counts if c.get("revisit")) / n,
        "crop_rate": sum(1 for c in counts if c.get("crop")) / n,
        "avg_final_pages": sum(len(r.get("final_pages") or []) for r in rows) / n,
        "final_coverage": (sum(coverage) / len(coverage)) if coverage else None,
        "avg_non_gold_final_pages": (sum(noise) / len(noise)) if noise else None,
        "accuracy_by_hop": accuracy_by_hop(rows),
    }


def accuracy_by_hop(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Judge accuracy per hop type (SCORE reports Single / Multi); unjudged rows are left out."""
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r.get("judge"), dict):
            continue
        group = out.setdefault(str(r.get("hop")), {"judged": 0, "correct": 0})
        group["judged"] += 1
        group["correct"] += int(r["judge"].get("correct") is True)
    for group in out.values():
        group["accuracy"] = group["correct"] / group["judged"]
    return dict(sorted(out.items()))


if __name__ == "__main__":
    main()
