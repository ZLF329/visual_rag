#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import random
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.agent import Agent
from src.config import load_config
from src.judge import judge_prediction_row, load_dotenv
from src.retriever import Retriever
from src.vlm import VLM


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Active-Clue-Graph trajectories and VISOR-style dynamic-chat SFT rows."
    )
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--config", default="config/eval_active_clue_graph_base_test500.yaml")
    parser.add_argument("--output-dir", default="outputs/active_graph_sft")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--index", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--judge", choices=["none", "deepseek"], default="none")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--judge-max-tokens", type=int, default=None)
    parser.add_argument("--require-judge-correct", action="store_true")
    parser.add_argument("--require-all-reference-pages", action="store_true")
    parser.add_argument("--stop-on-api-error", action="store_true")
    parser.add_argument("--target-kept", type=int, default=0, help="Stop submitting new samples after this many kept trajectories; 0 disables.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent samples to generate in one process. The retriever is shared and locked.",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    for env_file in config["models"]["vlm"].get("env_files", []):
        load_dotenv(env_file)
    if args.require_judge_correct and args.judge == "none":
        raise ValueError("--require-judge-correct requires --judge deepseek")
    device = args.device or config["runtime"]["device"]
    run_root = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(Path(args.dataset_file), start=args.start_index, limit=args.num_samples)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)

    raw_retriever = Retriever(
        model_path=config["models"]["retriever"]["name"],
        index_path=args.index or config["models"]["retriever"]["index_path"],
        device=device,
        dtype=config["runtime"]["dtype"],
        attn_implementation=config["runtime"].get("attn_implementation"),
    )
    workers = max(1, int(args.workers))
    retriever = LockedRetriever(raw_retriever) if workers > 1 else raw_retriever
    vlm = VLM(
        model_path=config["models"]["vlm"]["name"],
        device=device,
        max_tokens=config["models"]["vlm"]["max_tokens"],
        temperature=config["models"]["vlm"]["temperature"],
        dtype=config["runtime"]["dtype"],
        attn_implementation=config["runtime"].get("attn_implementation"),
        prompt_mode=config["models"]["vlm"].get("prompt_mode", "chat"),
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

    trajectories_path = run_root / "trajectories.jsonl"
    kept_trajectories_path = run_root / "kept_trajectories.jsonl"
    rejected_trajectories_path = run_root / "rejected_trajectories.jsonl"
    sft_path = run_root / "sft_dynamic_chat.jsonl"
    all_sft_path = run_root / "all_sft_dynamic_chat.jsonl"
    stats: dict[str, Any] = {
        "processed_samples": 0,
        "total_sft": 0,
        "total_all_sft": 0,
        "kept_samples": 0,
        "rejected_samples": 0,
        "reject_reasons": Counter(),
        "stopped_on_api_error": False,
        "stop_reason": "",
    }

    def submit(
        executor: ThreadPoolExecutor,
        futures: dict[Future[dict[str, Any]], tuple[int, dict[str, Any]]],
        local_idx: int,
        row: dict[str, Any],
    ) -> None:
        future = executor.submit(
            run_sample,
            row,
            local_idx=local_idx,
            absolute_idx=args.start_index + local_idx,
            run_root=run_root,
            vlm=vlm,
            retriever=retriever,
            top_k=config["agent"]["top_k"],
            max_iters=config["agent"]["max_iters"],
            bbox_frame=config["agent"].get("bbox_frame", "displayed_px"),
            judge=args.judge,
            judge_model=args.judge_model,
            judge_base_url=args.judge_base_url,
            judge_timeout=args.judge_timeout,
            judge_max_retries=args.judge_max_retries,
            judge_max_tokens=args.judge_max_tokens,
        )
        futures[future] = (local_idx, row)

    def record(result: dict[str, Any], local_idx: int) -> bool:
        should_stop = record_sample(
            result,
            local_idx=local_idx,
            total=len(rows),
            require_judge_correct=args.require_judge_correct,
            require_all_reference_pages=args.require_all_reference_pages,
            stop_on_api_error=args.stop_on_api_error,
            trajectories_path=trajectories_path,
            kept_trajectories_path=kept_trajectories_path,
            rejected_trajectories_path=rejected_trajectories_path,
            sft_path=sft_path,
            all_sft_path=all_sft_path,
            stats=stats,
        )
        if should_stop:
            return True
        if args.target_kept and stats["kept_samples"] >= args.target_kept:
            print(
                f"[sft-gen] reached target_kept={args.target_kept}; stopping submissions",
                flush=True,
            )
            return True
        return False

    if workers == 1:
        for local_idx, row in enumerate(rows):
            result = run_sample(
                row,
                local_idx=local_idx,
                absolute_idx=args.start_index + local_idx,
                run_root=run_root,
                vlm=vlm,
                retriever=retriever,
                top_k=config["agent"]["top_k"],
                max_iters=config["agent"]["max_iters"],
                bbox_frame=config["agent"].get("bbox_frame", "displayed_px"),
                judge=args.judge,
                judge_model=args.judge_model,
                judge_base_url=args.judge_base_url,
                judge_timeout=args.judge_timeout,
                judge_max_retries=args.judge_max_retries,
                judge_max_tokens=args.judge_max_tokens,
            )
            if record(result, local_idx):
                break
    else:
        next_idx = 0
        stop_submitting = False
        futures: dict[Future[dict[str, Any]], tuple[int, dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while next_idx < len(rows) and len(futures) < workers:
                submit(executor, futures, next_idx, rows[next_idx])
                next_idx += 1

            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    local_idx, row = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = build_runner_error_result(
                            row,
                            absolute_idx=args.start_index + local_idx,
                            error=exc,
                        )
                    if record(result, local_idx):
                        stop_submitting = True

                while not stop_submitting and next_idx < len(rows) and len(futures) < workers:
                    submit(executor, futures, next_idx, rows[next_idx])
                    next_idx += 1

    summary = {
        "run_root": str(run_root),
        "trajectories": str(trajectories_path),
        "kept_trajectories": str(kept_trajectories_path),
        "rejected_trajectories": str(rejected_trajectories_path),
        "sft_dynamic_chat": str(sft_path),
        "all_sft_dynamic_chat": str(all_sft_path),
        "num_samples": len(rows),
        "processed_samples": stats["processed_samples"],
        "kept_samples": stats["kept_samples"],
        "rejected_samples": stats["rejected_samples"],
        "reject_reasons": dict(stats["reject_reasons"]),
        "sft_calls": stats["total_sft"],
        "all_sft_calls": stats["total_all_sft"],
        "workers": workers,
        "stopped_on_api_error": stats["stopped_on_api_error"],
        "stop_reason": stats["stop_reason"],
        "filters": {
            "judge": args.judge,
            "require_judge_correct": args.require_judge_correct,
            "require_all_reference_pages": args.require_all_reference_pages,
            "stop_on_api_error": args.stop_on_api_error,
            "target_kept": args.target_kept,
        },
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


class LockedRetriever:
    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        self.lock = threading.Lock()

    def search(self, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            return self.retriever.search(*args, **kwargs)

    def search_labels(self, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            return self.retriever.search_labels(*args, **kwargs)


def run_sample(
    row: dict[str, Any],
    *,
    local_idx: int,
    absolute_idx: int,
    run_root: Path,
    vlm: VLM,
    retriever: Any,
    top_k: int,
    max_iters: int,
    bbox_frame: str = "displayed_px",
    judge: str,
    judge_model: str | None,
    judge_base_url: str | None,
    judge_timeout: float,
    judge_max_retries: int,
    judge_max_tokens: int | None,
) -> dict[str, Any]:
    sample_id = extract_sample_id(row, absolute_idx)
    query = extract_question(row)
    deck_name = extract_deck_name(row)
    sample_dir = run_root / "samples" / safe_name(sample_id)
    agent = Agent(
        vlm=vlm,
        retriever=retriever,
        top_k=top_k,
        max_iters=max_iters,
        bbox_frame=bbox_frame,
    )
    result = agent.run(query, output_dir=sample_dir, deck_name=deck_name)
    result["sample_id"] = sample_id
    result["source_row_index"] = absolute_idx
    result["local_row_index"] = local_idx
    result["gold_answer"] = extract_answer(row)
    result["evidence_pages"] = sorted(extract_reference_pages(row))
    if judge == "deepseek":
        try:
            result["judge"] = judge_prediction_row(
                result,
                model=judge_model,
                base_url=judge_base_url,
                timeout=judge_timeout,
                max_retries=judge_max_retries,
                max_tokens=judge_max_tokens,
            )
        except Exception as exc:
            result["judge_error"] = str(exc)
    result["retrieved_pages"] = sorted(extract_retrieved_pages(result))
    return result


def build_runner_error_result(
    row: dict[str, Any],
    *,
    absolute_idx: int,
    error: Exception,
) -> dict[str, Any]:
    sample_id = extract_sample_id(row, absolute_idx)
    return {
        "query": extract_question(row),
        "deck_name": extract_deck_name(row),
        "answer": "",
        "sample_id": sample_id,
        "source_row_index": absolute_idx,
        "gold_answer": extract_answer(row),
        "evidence_pages": sorted(extract_reference_pages(row)),
        "retrieved_pages": [],
        "trace": [{"step": "runner_error", "error": str(error)}],
        "terminated_by": "runner_error",
    }


def record_sample(
    result: dict[str, Any],
    *,
    local_idx: int,
    total: int,
    require_judge_correct: bool,
    require_all_reference_pages: bool,
    stop_on_api_error: bool,
    trajectories_path: Path,
    kept_trajectories_path: Path,
    rejected_trajectories_path: Path,
    sft_path: Path,
    all_sft_path: Path,
    stats: dict[str, Any],
) -> bool:
    stats["processed_samples"] += 1
    sample_id = result.get("sample_id")
    if stop_on_api_error and has_api_error(result):
        result["sft_keep"] = False
        result["sft_reject_reasons"] = ["api_error"]
        append_jsonl(trajectories_path, result)
        append_jsonl(rejected_trajectories_path, result)
        stats["rejected_samples"] += 1
        stats["reject_reasons"].update(["api_error"])
        stats["stopped_on_api_error"] = True
        stats["stop_reason"] = first_trace_error(result)
        print(
            f"[sft-gen] stopping on API error sample_id={sample_id}: "
            f"{first_trace_error(result)}",
            flush=True,
        )
        return True

    keep, reasons = should_keep_trajectory(
        result,
        require_judge_correct=require_judge_correct,
        require_all_reference_pages=require_all_reference_pages,
    )
    result["sft_keep"] = keep
    result["sft_reject_reasons"] = reasons
    append_jsonl(trajectories_path, result)

    calls = build_dynamic_chat_sft_calls(
        result,
        sample_id=str(sample_id),
    )
    for call in calls:
        append_jsonl(all_sft_path, call)
        if keep:
            append_jsonl(sft_path, call)
    stats["total_all_sft"] += len(calls)
    if keep:
        stats["kept_samples"] += 1
        stats["total_sft"] += len(calls)
        append_jsonl(kept_trajectories_path, result)
    else:
        stats["rejected_samples"] += 1
        stats["reject_reasons"].update(reasons)
        append_jsonl(rejected_trajectories_path, result)
    print(
        f"[sft-gen] {stats['processed_samples']}/{total} row={local_idx} "
        f"sample_id={sample_id} terminated_by={result.get('terminated_by')} "
        f"keep={keep} reasons={reasons} sft_calls={len(calls)}",
        flush=True,
    )
    return False


AGV2_ERROR_STEPS = {
    "policy_error",
    "update_graph_error",
    "invalid_bbox_no_target",
    "box_error",
    "runner_error",
}


def build_dynamic_chat_sft_calls(
    result: dict[str, Any],
    *,
    sample_id: str,
) -> list[dict[str, Any]]:
    """AGv2: each successful trace row (step == "turn") already carries the EXACT inference
    context in sft.messages/image_paths and the raw model response in sft.target — no prompt
    reconstruction. Error rows (AGV2_ERROR_STEPS) are skipped; trajectory-level rejection of
    episodes containing them happens in should_keep_trajectory."""
    calls: list[dict[str, Any]] = []
    for step in result.get("trace", []):
        if step.get("step") != "turn":
            continue
        sft = step.get("sft") or {}
        target = str(sft.get("target") or "").strip()
        messages = sft.get("messages") or []
        if not target or not messages:
            continue
        image_paths = [str(p) for p in sft.get("image_paths") or []]
        call = {
            "id": f"{sample_id}:turn:{step.get('iter')}",
            "sample_id": sample_id,
            "iter": step.get("iter"),
            "step": "turn",
            "action": step.get("action"),
            "active_node_id": step.get("active_node_id"),
            "messages": messages,
            "image_paths": image_paths,
            "target": target,
        }
        call["conversations"] = to_qwenvl_conversations(messages, target)
        calls.append(call)
    return calls


def should_keep_trajectory(
    result: dict[str, Any],
    *,
    require_judge_correct: bool,
    require_all_reference_pages: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    # AGv2 format hygiene: never train on episodes containing protocol violations or box
    # errors, even if the final answer happens to be judged correct.
    if str(result.get("terminated_by") or "") != "answer":
        reasons.append(f"terminated_by:{result.get('terminated_by')}")
    error_steps = sorted(
        {
            str(step.get("step"))
            for step in result.get("trace") or []
            if str(step.get("step")) in AGV2_ERROR_STEPS
        }
    )
    if error_steps:
        reasons.append("error_steps:" + ",".join(error_steps))
    if require_judge_correct:
        judge = result.get("judge")
        if not isinstance(judge, dict):
            reasons.append("judge_missing")
        elif judge.get("correct") is not True:
            reasons.append("judge_incorrect")
    if require_all_reference_pages:
        reference_pages = set(result.get("evidence_pages") or [])
        retrieved_pages = set(result.get("retrieved_pages") or [])
        if not reference_pages:
            reasons.append("reference_pages_missing")
        elif not reference_pages.issubset(retrieved_pages):
            missing = ",".join(str(page) for page in sorted(reference_pages - retrieved_pages))
            reasons.append(f"missing_reference_pages:{missing}")
    return not reasons, reasons


def extract_reference_pages(row: dict[str, Any]) -> set[int]:
    value = first_present(
        row,
        ("evidence_pages", "reference_pages", "gold_pages", "page_ids", "pages", "answer_pages"),
        [],
    )
    if isinstance(value, str):
        return {int(item) for item in re.findall(r"\d+", value)}
    if isinstance(value, (int, float)):
        return {int(value)}
    out: set[int] = set()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                item = first_present(item, ("page", "page_num", "page_id", "index"), None)
            try:
                out.add(int(item))
            except (TypeError, ValueError):
                continue
    return out


def extract_retrieved_pages(result: dict[str, Any]) -> set[int]:
    pages: set[int] = set()
    for step in result.get("trace") or []:
        observation = step.get("observation") or {}
        for image in observation.get("images") or []:
            for key in ("page_label", "source_image_id", "image_id"):
                page = page_number_from_label(image.get(key))
                if page is not None:
                    pages.add(page)
    return pages


def page_number_from_label(value: Any) -> int | None:
    match = re.search(r"page[_-]?(\d+)", str(value or ""), flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def has_api_error(result: dict[str, Any]) -> bool:
    error_text = first_trace_error(result)
    # Parse/protocol failures are model behavior, never infrastructure errors — and their
    # message embeds the model's own text (last_output='...'), which must not be scanned
    # for API-error markers (a <think> containing e.g. "insufficient" would false-match).
    if error_text.startswith("failed to parse"):
        return False
    error_text = error_text.split("last_output='", 1)[0].lower()
    markers = (
        "api ",
        "quota",
        "balance",
        "insufficient",
        "rate limit",
        "rate_limit",
        "too many requests",
        "billing",
        "invalid api key",
        "model not found",
        "no permission",
        "access denied",
        "forbidden",
    )
    return any(marker in error_text for marker in markers)


def first_trace_error(result: dict[str, Any]) -> str:
    for step in result.get("trace") or []:
        error = str(step.get("error") or "").strip()
        if error:
            return error
    return str(result.get("judge_error") or "")


def to_qwenvl_conversations(messages: list[dict[str, str]], target: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    role_map = {"system": "system", "user": "human", "assistant": "gpt"}
    for message in messages:
        out.append(
            {
                "from": role_map.get(message["role"], message["role"]),
                "value": message["content"],
            }
        )
    out.append({"from": "gpt", "value": target})
    return out


def read_jsonl(path: Path, *, start: int, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if seen < start:
                seen += 1
                continue
            rows.append(json.loads(line))
            seen += 1
            if len(rows) >= limit:
                break
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_present(row: dict[str, Any], names: tuple[str, ...], default: Any = "") -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def extract_question(row: dict[str, Any]) -> str:
    return str(first_present(row, ("query", "question", "prompt", "input", "problem"))).strip()


def extract_answer(row: dict[str, Any]) -> str:
    value = first_present(row, ("answer", "answers", "target", "label", "response"))
    return " | ".join(str(item) for item in value) if isinstance(value, list) else str(value)


def extract_deck_name(row: dict[str, Any]) -> str:
    return str(first_present(row, ("deck_name", "deck_id", "doc_id", "document_id", "pdf_id"))).strip()


def extract_sample_id(row: dict[str, Any], idx: int) -> str:
    return str(first_present(row, ("id", "qid", "question_id", "uid", "eval_id"), idx))


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))[:120]


if __name__ == "__main__":
    main()
