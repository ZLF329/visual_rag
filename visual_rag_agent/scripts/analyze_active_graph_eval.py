#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Active-Clue-Graph eval predictions and optional judge results."
    )
    parser.add_argument("paths", nargs="+", help="Prediction JSONL files, judged JSONL files, or run directories.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--examples-per-bucket", type=int, default=8)
    args = parser.parse_args()

    files = expand_inputs([Path(path) for path in args.paths])
    rows = load_rows(files)
    report = analyze_rows(rows, examples_per_bucket=args.examples_per_bucket)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


def expand_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("**/judged_predictions.jsonl")))
            files.extend(sorted(path.glob("**/predictions.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            files.extend(sorted(Path().glob(str(path))))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def load_rows(files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_analysis_source_file"] = str(path)
                rows.append(row)
    return rows


def analyze_rows(rows: list[dict[str, Any]], *, examples_per_bucket: int) -> dict[str, Any]:
    termination = Counter(str(row.get("terminated_by") or "unknown") for row in rows)
    policy_actions: Counter[str] = Counter()
    graph_decisions: Counter[str] = Counter()
    error_messages: dict[str, Counter[str]] = defaultdict(Counter)
    trace_lengths: list[int] = []
    search_queries = 0
    crop_actions = 0

    for row in rows:
        trace = row.get("trace") or []
        trace_lengths.append(len(trace))
        for step in trace:
            if step.get("step") == "policy":
                result = step.get("result") or {}
                action_type = result.get("type")
                if action_type:
                    policy_actions[str(action_type)] += 1
            if step.get("step") == "update_graph":
                decision = step.get("decision") or {}
                decision_type = decision.get("type")
                if decision_type:
                    graph_decisions[str(decision_type)] += 1
            if step.get("step") == "search":
                search_queries += 1
            if step.get("step") == "crop":
                crop_actions += 1
            if str(step.get("step", "")).endswith("_error"):
                error_messages[str(step.get("step"))][compact_error(step.get("error"))] += 1

    judged = [row for row in rows if isinstance(row.get("judge"), dict)]
    correct = [row for row in judged if row["judge"].get("correct") is True]
    judge_errors = [row for row in rows if row.get("judge_error")]
    judge_by_termination: dict[str, dict[str, Any]] = {}
    if judged:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in judged:
            grouped[str(row.get("terminated_by") or "unknown")].append(row)
        for key, items in sorted(grouped.items()):
            n_correct = sum(1 for item in items if item["judge"].get("correct") is True)
            judge_by_termination[key] = {
                "judged": len(items),
                "correct": n_correct,
                "accuracy": n_correct / len(items) if items else 0.0,
            }

    examples = {
        "max_iters": collect_examples(rows, termination="max_iters", limit=examples_per_bucket),
        "policy_error": collect_examples(rows, termination="policy_error", limit=examples_per_bucket),
        "crop_error": collect_examples(rows, termination="crop_error", limit=examples_per_bucket),
        "update_graph_error": collect_examples(rows, termination="update_graph_error", limit=examples_per_bucket),
        "judge_wrong": collect_wrong_examples(rows, limit=examples_per_bucket),
    }

    return {
        "num_rows": len(rows),
        "termination_counts": dict(termination),
        "max_iters_ratio": termination.get("max_iters", 0) / len(rows) if rows else 0.0,
        "judge": {
            "judged": len(judged),
            "correct": len(correct),
            "errors": len(judge_errors),
            "accuracy": len(correct) / len(judged) if judged else None,
        },
        "judge_by_termination": judge_by_termination,
        "policy_action_distribution": dict(policy_actions),
        "graph_decision_distribution": dict(graph_decisions),
        "search_queries": search_queries,
        "crop_observations": crop_actions,
        "mean_trace_steps": sum(trace_lengths) / len(trace_lengths) if trace_lengths else 0.0,
        "error_messages": {
            step: dict(counter.most_common(10)) for step, counter in sorted(error_messages.items())
        },
        "examples": examples,
    }


def collect_examples(
    rows: list[dict[str, Any]],
    *,
    termination: str,
    limit: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        if row.get("terminated_by") != termination:
            continue
        examples.append(example_summary(row))
        if len(examples) >= limit:
            break
    return examples


def collect_wrong_examples(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        judge = row.get("judge")
        if not isinstance(judge, dict) or judge.get("correct") is True:
            continue
        item = example_summary(row)
        item["judge_rationale"] = judge.get("rationale")
        item["normalized_prediction"] = judge.get("normalized_prediction")
        item["normalized_gold"] = judge.get("normalized_gold")
        examples.append(item)
        if len(examples) >= limit:
            break
    return examples


def example_summary(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("trace") or []
    last_error = ""
    for step in reversed(trace):
        if step.get("error"):
            last_error = compact_error(step.get("error"), limit=500)
            break
    return {
        "sample_id": row.get("sample_id"),
        "terminated_by": row.get("terminated_by"),
        "question": row.get("query"),
        "prediction": row.get("answer"),
        "gold_answer": row.get("gold_answer"),
        "last_error": last_error,
        "last_steps": [
            {
                "iter": step.get("iter"),
                "step": step.get("step"),
                "action": (step.get("result") or {}).get("type"),
                "query": step.get("query"),
                "decision": (step.get("decision") or {}).get("type"),
                "error": compact_error(step.get("error")),
            }
            for step in trace[-5:]
        ],
    }


def compact_error(value: Any, *, limit: int = 220) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:limit]


if __name__ == "__main__":
    main()
