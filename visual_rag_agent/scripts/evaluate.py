#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from src.agent import Agent, is_fatal_model_error
from src.config import load_config
from src.judge import add_judge_summary, judge_prediction_row, load_dotenv
from src.retriever import Retriever, ColQwenRetriever
from src.summary_baseline import AgenticSummaryBaseline
from src.vlm import VLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the agent or agentic-summary baseline.")
    parser.add_argument("--dataset-file", required=True, help="JSONL file with SlideVQA examples.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=0, help="Skip this many JSONL rows before evaluating.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="outputs/runs")
    parser.add_argument("--baseline", choices=["agent", "agentic_summary"], default="agent")
    parser.add_argument("--index", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--judge", choices=["none", "deepseek"], default="none")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--judge-max-tokens", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1, help="parallel rollouts via ThreadPool (1=sequential)")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    for env_file in config["models"]["vlm"].get("env_files", []):
        load_dotenv(env_file)
    device = args.device or config["runtime"]["device"]
    preflight_runtime(device, config["runtime"].get("attn_implementation"))
    index_path = args.index or config["models"]["retriever"]["index_path"]
    run_root = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    examples = read_jsonl(Path(args.dataset_file), limit=args.num_samples, start=args.start_index)
    _rname = config["models"]["retriever"]["name"]
    _rkwargs = dict(
        model_path=_rname,
        index_path=index_path,
        device=device,
        dtype=config["runtime"]["dtype"],
        attn_implementation=config["runtime"].get("attn_implementation"),
    )
    if "colqwen" in str(_rname).lower():
        retriever = ColQwenRetriever(**_rkwargs)
    else:
        retriever = Retriever(**_rkwargs)
    vlm = VLM(
        model_path=config["models"]["vlm"]["name"],
        device=device,
        max_tokens=config["models"]["vlm"]["max_tokens"],
        temperature=config["models"]["vlm"]["temperature"],
        dtype=config["runtime"]["dtype"],
        attn_implementation=config["runtime"].get("attn_implementation"),
        prompt_mode=config["models"]["vlm"].get("prompt_mode", "chat"),
        adapter_path=config["models"]["vlm"].get("adapter_path"),
        provider=config["models"]["vlm"].get("provider", "qwen"),
        api_base_url=config["models"]["vlm"].get("api_base_url"),
        api_key_env=config["models"]["vlm"].get("api_key_env", "MIMO_API_KEY"),
        api_timeout=float(config["models"]["vlm"].get("api_timeout", 180.0)),
        api_max_retries=int(config["models"]["vlm"].get("api_max_retries", 2)),
        api_image_quality=int(config["models"]["vlm"].get("api_image_quality", 85)),
        api_extra_body=config["models"]["vlm"].get("api_extra_body") or {},
        policy_output_format=config["models"]["vlm"].get("policy_output_format", "tags"),
        api_policy_response_format=config["models"]["vlm"].get("api_policy_response_format"),
    )

    predictions_path = run_root / "predictions.jsonl"
    prediction_rows: list[dict[str, Any]] = []
    if args.baseline == "agentic_summary":
        baseline = AgenticSummaryBaseline(
            vlm=vlm,
            retriever=retriever,
            top_k=config["agent"]["top_k"],
            max_iters=config["agent"]["max_iters"],
        )
        for idx, row in enumerate(examples):
            sample_id = extract_sample_id(row, idx)
            query = extract_question(row)
            deck_name = extract_deck_name(row)
            try:
                result = baseline.run(query, output_dir=run_root / str(sample_id), deck_name=deck_name)
            except Exception as exc:
                handle_sample_exception(exc, idx=idx, total=len(examples), sample_id=sample_id)
            record_prediction(
                result,
                source_row=row,
                sample_id=sample_id,
                args=args,
                prediction_rows=prediction_rows,
                predictions_path=predictions_path,
            )
            print(f"[eval] step={idx + 1}/{len(examples)} sample_id={sample_id} terminated_by={result.get('terminated_by')}", flush=True)
    else:
        import threading
        from concurrent.futures import ThreadPoolExecutor
        _conc = max(1, int(getattr(args, "concurrency", 1) or 1))
        # vLLM (openai_compatible) handles concurrent requests; the CUDA retriever must be serialized.
        _ret = retriever
        if _conc > 1:
            class _LockedRetriever:
                def __init__(self, inner):
                    self._inner = inner
                    self._lock = threading.Lock()
                def __getattr__(self, name):
                    attr = getattr(self._inner, name)
                    if callable(attr):
                        def _w(*a, **k):
                            with self._lock:
                                return attr(*a, **k)
                        return _w
                    return attr
            _ret = _LockedRetriever(retriever)
        _rec_lock = threading.Lock()
        _done = [0]

        def _run_one(pair):
            idx, row = pair
            sample_id = extract_sample_id(row, idx)
            query = extract_question(row)
            deck_name = extract_deck_name(row)
            agent = Agent(
                vlm=vlm,
                retriever=_ret,
                top_k=config["agent"]["top_k"],
                max_iters=config["agent"]["max_iters"],
            )
            try:
                result = agent.run(query, output_dir=run_root / str(sample_id), deck_name=deck_name)
            except Exception as exc:
                handle_sample_exception(exc, idx=idx, total=len(examples), sample_id=sample_id)
                return
            with _rec_lock:
                record_prediction(
                    result,
                    source_row=row,
                    sample_id=sample_id,
                    args=args,
                    prediction_rows=prediction_rows,
                    predictions_path=predictions_path,
                )
                _done[0] += 1
                print(f"[eval] step={_done[0]}/{len(examples)} sample_id={sample_id} terminated_by={result.get('terminated_by')}", flush=True)

        if _conc > 1:
            with ThreadPoolExecutor(max_workers=_conc) as _ex:
                list(_ex.map(_run_one, list(enumerate(examples))))
        else:
            for _pair in enumerate(examples):
                _run_one(_pair)

    write_jsonl(predictions_path, prediction_rows)
    summary = summarize(prediction_rows, baseline=args.baseline)
    add_judge_summary(summary, prediction_rows)
    with (run_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"run_root": str(run_root), "summary": summary}, ensure_ascii=False, indent=2))


def read_jsonl(path: Path, limit: int | None = None, start: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if seen < start:
                seen += 1
                continue
            rows.append(json.loads(line))
            seen += 1
            if limit is not None and len(rows) >= limit:
                break
    return rows


def preflight_runtime(device: str, attn_implementation: str | None) -> None:
    if device != "cuda":
        return
    import torch

    if torch.cuda.is_available():
        return
    detail = "CUDA device is not visible to torch"
    if attn_implementation == "flash_attention_2":
        detail += "; flash_attention_2 cannot run on CPU"
    raise RuntimeError(
        f"{detail}. Start this eval on a GPU instance or pass a CPU-capable config."
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def short_error(exc: Exception, limit: int = 240) -> str:
    text = str(exc).strip() or type(exc).__name__
    first_line = text.splitlines()[0] if text else type(exc).__name__
    return first_line[:limit]


def handle_sample_exception(exc: Exception, *, idx: int, total: int, sample_id: str) -> None:
    if is_fatal_model_error(exc):
        print(
            f"[eval] fatal step={idx + 1}/{total} sample_id={sample_id} error={short_error(exc)}",
            flush=True,
        )
    raise exc


def record_prediction(
    result: dict[str, Any],
    *,
    source_row: dict[str, Any],
    sample_id: str,
    args: argparse.Namespace,
    prediction_rows: list[dict[str, Any]],
    predictions_path: Path,
) -> None:
    result["sample_id"] = sample_id
    result["gold_answer"] = extract_answer(source_row)
    if args.judge == "deepseek":
        try:
            result["judge"] = judge_prediction_row(
                result,
                model=args.judge_model,
                base_url=args.judge_base_url,
                timeout=args.judge_timeout,
                max_retries=args.judge_max_retries,
                max_tokens=args.judge_max_tokens,
            )
        except Exception as exc:
            result["judge_error"] = str(exc)
            print(f"[eval] judge_error sample_id={sample_id}: {short_error(exc)}", flush=True)
    prediction_rows.append(result)
    append_jsonl(predictions_path, result)


def extract_question(row: dict[str, Any]) -> str:
    return str(first_present(row, ("query", "question", "prompt", "input", "problem"), "")).strip()


def extract_answer(row: dict[str, Any]) -> str:
    value = first_present(row, ("answer", "answers", "target", "label", "response"), "")
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value)


def extract_deck_name(row: dict[str, Any]) -> str:
    return str(
        first_present(row, ("deck_name", "deck_id", "doc_id", "document_id", "pdf_id"), "")
    ).strip()


def extract_sample_id(row: dict[str, Any], idx: int) -> str:
    return str(first_present(row, ("id", "qid", "question_id", "uid", "eval_id"), idx))


def first_present(row: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def summarize(rows: list[dict[str, Any]], *, baseline: str) -> dict[str, Any]:
    if baseline == "agentic_summary":
        return summarize_agentic_summary(rows)

    iters: list[int] = []
    termination_counts: dict[str, int] = {}
    decision_distribution = {"yes": 0, "partial": 0, "no": 0}
    graph_decision_distribution = {"ACCEPT": 0, "REJECT": 0, "EXPAND": 0}
    policy_action_distribution = {"SEARCH": 0, "CROP": 0, "UPDATE_GRAPH": 0, "ANSWER": 0}
    failure_type_counts: dict[str, int] = {}
    visual_budgets: list[int] = []
    graph_nodes: list[int] = []
    search_queries = 0
    warning_informed_queries = 0

    for row in rows:
        trace = row.get("trace", [])
        memory = row.get("memory", {})
        graph = row.get("graph", {})
        iters.append(int(memory.get("iter", 0)))
        graph_nodes.append(len(graph.get("nodes", {})) if isinstance(graph, dict) else 0)
        terminated_by = row.get("terminated_by", "unknown")
        termination_counts[terminated_by] = termination_counts.get(terminated_by, 0) + 1
        prior_warnings = 0
        for step in trace:
            if step.get("step") == "policy":
                action = (step.get("result") or {}).get("type")
                if action in policy_action_distribution:
                    policy_action_distribution[action] += 1
            if step.get("step") == "search":
                search_queries += 1
                if prior_warnings > 0:
                    warning_informed_queries += 1
            if step.get("step") == "update_graph":
                decision_type = (step.get("decision") or {}).get("type")
                if decision_type in graph_decision_distribution:
                    graph_decision_distribution[decision_type] += 1
            if step.get("step") == "analyse":
                result = step.get("result", {})
                decision = result.get("judge") or result.get("decision")
                if decision in decision_distribution:
                    decision_distribution[decision] += 1
                if decision == "no":
                    failure_type = result.get("failure_type", "unknown")
                    failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
                    prior_warnings += 1
        visual_budgets.append(estimate_visual_budget(memory))

    return {
        "num_samples": len(rows),
        "baseline": baseline,
        "mean_iters": statistics.mean(iters) if iters else 0,
        "median_iters": statistics.median(iters) if iters else 0,
        "termination_counts": termination_counts,
        "decision_distribution": decision_distribution,
        "graph_decision_distribution": graph_decision_distribution,
        "policy_action_distribution": policy_action_distribution,
        "mean_graph_nodes": statistics.mean(graph_nodes) if graph_nodes else 0,
        "warning_utilization": {
            "search_queries": search_queries,
            "warning_informed_queries": warning_informed_queries,
            "fraction": warning_informed_queries / search_queries if search_queries else 0,
        },
        "avg_visual_pixels_at_answer": statistics.mean(visual_budgets) if visual_budgets else 0,
        "failure_type_counts": failure_type_counts,
        "accuracy": "not_computed_use_external_llm_judge",
    }


def summarize_agentic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    iters: list[int] = []
    n_summaries: list[int] = []
    termination_counts: dict[str, int] = {}
    search_queries = 0
    summarise_calls = 0
    for row in rows:
        memory = row.get("summary_memory", {})
        trace = row.get("trace", [])
        iters.append(int(memory.get("iter", 0)))
        n_summaries.append(len(memory.get("summaries", [])))
        terminated_by = row.get("terminated_by", "unknown")
        termination_counts[terminated_by] = termination_counts.get(terminated_by, 0) + 1
        for step in trace:
            if step.get("step") == "baseline_search":
                search_queries += 1
            if step.get("step") == "baseline_summarise":
                summarise_calls += 1
    return {
        "num_samples": len(rows),
        "baseline": "agentic_summary",
        "mean_iters": statistics.mean(iters) if iters else 0,
        "median_iters": statistics.median(iters) if iters else 0,
        "termination_counts": termination_counts,
        "mean_page_summaries": statistics.mean(n_summaries) if n_summaries else 0,
        "search_queries": search_queries,
        "summarise_calls": summarise_calls,
        "accuracy": "not_computed_use_external_llm_judge",
    }


def estimate_visual_budget(memory: dict[str, Any]) -> int:
    total = 0
    for finding in memory.get("confirmed", []):
        total += int(finding.get("image_ref", {}).get("pixel_budget", 0))
    for finding in memory.get("partials", []):
        total += int(finding.get("image_ref", {}).get("pixel_budget", 0))
    return total


if __name__ == "__main__":
    main()
