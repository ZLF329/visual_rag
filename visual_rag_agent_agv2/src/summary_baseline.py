from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

from src.agent import is_fatal_model_error
from src.image_utils import resize_to_pixel_count


@dataclass
class SummaryEntry:
    iter: int
    page_label: str
    found_via_query: str
    summary: str
    visible_text: list[str]


@dataclass
class RecentImageEntry:
    iter: int
    page_label: str
    image_path: str | None
    image: Image.Image

    def as_trace_dict(self) -> dict[str, Any]:
        return {
            "iter": self.iter,
            "page_label": self.page_label,
            "image_path": self.image_path,
        }


class AgenticSummaryMemory:
    def __init__(
        self,
        *,
        original_query: str,
        image_dir: Path | None = None,
        recent_rounds: int = 2,
        recent_image_pixels: int = 200_000,
    ) -> None:
        self.original_query = original_query
        self.image_dir = image_dir
        self.recent_rounds = recent_rounds
        self.recent_image_pixels = recent_image_pixels
        self.iter = 0
        self.summaries: list[SummaryEntry] = []
        self.recent_images: deque[RecentImageEntry] = deque()
        self.issued_queries: list[str] = []

    def add_summary(
        self,
        *,
        page_label: str,
        search_query: str,
        summary: str,
        visible_text: list[str],
        image: Image.Image,
    ) -> None:
        self.summaries.append(
            SummaryEntry(
                iter=self.iter,
                page_label=page_label,
                found_via_query=search_query,
                summary=summary,
                visible_text=visible_text,
            )
        )
        resized = resize_to_pixel_count(image, self.recent_image_pixels)
        image_path = self._store_recent_image(resized, page_label=page_label)
        self.recent_images.append(
            RecentImageEntry(
                iter=self.iter,
                page_label=page_label,
                image_path=str(image_path) if image_path else None,
                image=resized.copy(),
            )
        )
        self._trim_recent_images()

    def context_for_search(self) -> str:
        if not self.summaries:
            return "None yet."
        lines: list[str] = []
        for idx, entry in enumerate(self.summaries, start=1):
            visible = "; ".join(entry.visible_text[:5]) if entry.visible_text else "None"
            lines.append(
                f"[{idx}] iter={entry.iter} page={entry.page_label} "
                f"query=\"{entry.found_via_query}\"\n"
                f"summary: {entry.summary}\n"
                f"visible_text: {visible}"
            )
        return "\n\n".join(lines)

    def recent_image_objects(self) -> list[Image.Image]:
        return [entry.image for entry in self.recent_images]

    def recent_image_labels_text(self) -> str:
        if not self.recent_images:
            return "None"
        return "\n".join(
            f"  - iter {entry.iter}: {entry.page_label}" for entry in self.recent_images
        )

    def as_serializable(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "iter": self.iter,
            "summaries": [asdict(entry) for entry in self.summaries],
            "recent_images": [entry.as_trace_dict() for entry in self.recent_images],
            "issued_queries": list(self.issued_queries),
            "recent_rounds": self.recent_rounds,
            "recent_image_pixels": self.recent_image_pixels,
        }

    def _trim_recent_images(self) -> None:
        min_iter = self.iter - self.recent_rounds + 1
        while self.recent_images and self.recent_images[0].iter < min_iter:
            self.recent_images.popleft()

    def _store_recent_image(self, image: Image.Image, *, page_label: str) -> Path | None:
        if self.image_dir is None:
            return None
        self.image_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(ch if ch.isalnum() else "_" for ch in page_label)[:80]
        out = self.image_dir / f"iter_{self.iter:02d}_{safe_label}_{self.recent_image_pixels // 1000}k.png"
        image.save(out)
        return out


class AgenticSummaryBaseline:
    def __init__(
        self,
        *,
        vlm: Any,
        retriever: Any,
        top_k: int = 1,
        max_iters: int = 5,
        recent_rounds: int = 2,
        recent_image_pixels: int = 200_000,
    ) -> None:
        self.vlm = vlm
        self.retriever = retriever
        self.top_k = top_k
        self.max_iters = max_iters
        self.recent_rounds = recent_rounds
        self.recent_image_pixels = recent_image_pixels

    def run(
        self,
        query: str,
        output_dir: str | Path | None = None,
        deck_name: str | None = None,
    ) -> dict[str, Any]:
        run_dir = Path(output_dir) if output_dir is not None else None
        image_dir = run_dir / "images" if run_dir is not None else None
        memory = AgenticSummaryMemory(
            original_query=query,
            image_dir=image_dir,
            recent_rounds=self.recent_rounds,
            recent_image_pixels=self.recent_image_pixels,
        )
        trace: list[dict[str, Any]] = []
        decision: Any | None = None

        while memory.iter < self.max_iters:
            try:
                decision = self.vlm.summary_baseline_decide(
                    original_query=query,
                    summaries_context=memory.context_for_search(),
                    recent_images=memory.recent_image_objects(),
                    recent_image_labels=memory.recent_image_labels_text(),
                )
                trace.append(
                    {
                        "iter": memory.iter,
                        "step": "baseline_decide",
                        "result": to_plain(decision),
                        "recent_image_labels": memory.recent_image_labels_text(),
                    }
                )
            except Exception as exc:
                if is_fatal_model_error(exc):
                    raise
                trace.append(
                    {
                        "iter": memory.iter,
                        "step": "baseline_decide_error",
                        "error": str(exc),
                    }
                )
                break

            if decision.action == "answer":
                break

            search_query = decision.search_query or query
            memory.issued_queries.append(search_query)

            try:
                try:
                    retrieved = self.retriever.search(
                        search_query,
                        top_k=self.top_k,
                        deck_name=deck_name,
                    )
                except TypeError:
                    retrieved = self.retriever.search(search_query, top_k=self.top_k)
                trace.append(
                    {
                        "iter": memory.iter,
                        "step": "baseline_search",
                        "query": search_query,
                        "deck_name": deck_name,
                        "n_images": len(retrieved),
                        "pages": [page_label for _, page_label in retrieved],
                    }
                )
            except Exception as exc:
                if is_fatal_model_error(exc):
                    raise
                trace.append(
                    {
                        "iter": memory.iter,
                        "step": "baseline_search_error",
                        "query": search_query,
                        "error": str(exc),
                    }
                )
                memory.iter += 1
                continue

            for image, page_label in retrieved:
                try:
                    summary = self.vlm.summary_baseline_summarise_image(
                        image=image,
                        original_query=query,
                        search_query=search_query,
                        summaries_context=memory.context_for_search(),
                        page_label=page_label,
                    )
                    memory.add_summary(
                        page_label=page_label,
                        search_query=search_query,
                        summary=summary.summary,
                        visible_text=summary.visible_text,
                        image=image,
                    )
                    trace.append(
                        {
                            "iter": memory.iter,
                            "step": "baseline_summarise",
                            "page": page_label,
                            "result": to_plain(summary),
                        }
                    )
                except Exception as exc:
                    if is_fatal_model_error(exc):
                        raise
                    trace.append(
                        {
                            "iter": memory.iter,
                            "step": "baseline_summarise_error",
                            "page": page_label,
                            "error": str(exc),
                        }
                    )

            memory.iter += 1

        answer = decision.answer.strip() if decision is not None and decision.action == "answer" and decision.answer else ""

        terminated_by = (
            "decide" if decision is not None and decision.action == "answer" else "max_iters"
        )
        if trace:
            last_step = str(trace[-1].get("step", ""))
            if last_step == "baseline_decide_error":
                terminated_by = last_step

        output = {
            "query": query,
            "answer": answer,
            "summary_memory": memory.as_serializable(),
            "trace": trace,
            "terminated_by": terminated_by,
            "baseline": "agentic_summary",
        }
        if run_dir is not None:
            write_agentic_summary_artifacts(run_dir, output)
        return output


def to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def write_agentic_summary_artifacts(run_dir: Path, output: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "query.txt").write_text(output["query"], encoding="utf-8")
    (run_dir / "answer.txt").write_text(output.get("answer", ""), encoding="utf-8")
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in output.get("trace", []):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (run_dir / "summary_memory.json").open("w", encoding="utf-8") as f:
        json.dump(output.get("summary_memory", {}), f, ensure_ascii=False, indent=2)
