from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.memory import Memory
from src.schemas import DecideResult


def is_fatal_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    fatal_markers = (
        "cuda out of memory",
        "cuda error: out of memory",
        "device-side assert",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
    )
    return any(marker in message for marker in fatal_markers)


class Agent:
    def __init__(self, vlm: Any, retriever: Any, top_k: int = 1, max_iters: int = 5) -> None:
        self.vlm = vlm
        self.retriever = retriever
        self.top_k = top_k
        self.max_iters = max_iters

    def run(self, query: str, output_dir: str | Path | None = None) -> dict[str, Any]:
        run_dir = Path(output_dir) if output_dir is not None else None
        image_dir = run_dir / "images" if run_dir is not None else None
        memory = Memory(original_query=query, image_dir=image_dir)
        trace: list[dict[str, Any]] = []
        decision: DecideResult | None = None

        while memory.iter < self.max_iters:
            try:
                decision = self.vlm.decide_next(
                    original_query=query,
                    memory=memory,
                    warnings_summary=memory.context_for_decide(),
                )
                trace.append(
                    {
                        "iter": memory.iter,
                        "step": "decide",
                        "result": to_plain(decision),
                    }
                )
            except Exception as exc:
                if is_fatal_model_error(exc):
                    raise
                trace.append(
                    {
                        "iter": memory.iter,
                        "step": "decide_error",
                        "error": str(exc),
                    }
                )
                break

            if decision.action == "answer":
                break

            search_query = decision.content or query
            try:
                retrieved = self.retriever.search(search_query, top_k=self.top_k)
            except Exception as exc:
                if is_fatal_model_error(exc):
                    raise
                trace.append(
                    {
                        "iter": memory.iter,
                        "step": "search_error",
                        "query": search_query,
                        "error": str(exc),
                    }
                )
                memory.add_empty_search_warning(search_query)
                memory.iter += 1
                continue

            trace.append(
                {
                    "iter": memory.iter,
                    "step": "search",
                    "query": search_query,
                    "n_images": len(retrieved),
                    "pages": [page_label for _, page_label in retrieved],
                }
            )

            if not retrieved:
                memory.add_empty_search_warning(search_query)
                memory.iter += 1
                continue

            for image, page_label in retrieved:
                try:
                    result = self.vlm.analyse(
                        image=image,
                        original_query=query,
                        memory_context=memory.context_for_analyse(),
                    )
                    memory.write(result, image, search_query, page_label=page_label)
                    trace.append(
                        {
                            "iter": memory.iter,
                            "step": "analyse",
                            "page": page_label,
                            "result": to_plain(result),
                        }
                    )
                    memory.append_consolidated_summary(result.summary, search_query=search_query)
                    if result.judge == "no":
                        trace.append(
                            {
                                "iter": memory.iter,
                                "step": "memory_update_skipped",
                                "reason": "judge_no",
                            }
                        )
                    else:
                        try:
                            memory_update = self.vlm.update_memory(
                                original_query=query,
                                memory=memory,
                                latest_analysis=result,
                                max_new_tokens=1000,
                            )
                            memory.update_consolidated(memory_update)
                            trace.append(
                                {
                                    "iter": memory.iter,
                                    "step": "memory_update",
                                    "result": to_plain(memory_update),
                                }
                            )
                        except Exception as exc:
                            if is_fatal_model_error(exc):
                                raise
                            trace.append(
                                {
                                    "iter": memory.iter,
                                    "step": "memory_update_error",
                                    "error": str(exc),
                                }
                            )
                except Exception as exc:
                    if is_fatal_model_error(exc):
                        raise
                    trace.append(
                        {
                            "iter": memory.iter,
                            "step": "analyse_error",
                            "page": page_label,
                            "error": str(exc),
                        }
                    )

            memory.iter += 1

        final = decision.content.strip() if decision is not None and decision.action == "answer" and decision.content else ""

        terminated_by = (
            "decide" if decision is not None and decision.action == "answer" else "max_iters"
        )
        if trace:
            last_step = str(trace[-1].get("step", ""))
            if last_step == "decide_error":
                terminated_by = last_step

        output = {
            "query": query,
            "answer": final,
            "memory": memory.as_serializable(),
            "trace": trace,
            "terminated_by": terminated_by,
        }
        if run_dir is not None:
            write_run_artifacts(run_dir, output)
        return output


def to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def write_run_artifacts(run_dir: Path, output: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "query.txt").write_text(output["query"], encoding="utf-8")
    (run_dir / "answer.txt").write_text(output.get("answer", ""), encoding="utf-8")
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in output.get("trace", []):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (run_dir / "memory_final.json").open("w", encoding="utf-8") as f:
        json.dump(output.get("memory", {}), f, ensure_ascii=False, indent=2)
