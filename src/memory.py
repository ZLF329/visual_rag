from __future__ import annotations

import hashlib
import io
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.image_utils import resize_to_pixel_count
from src.schemas import (
    AnalyseResult,
    ConfirmedFinding,
    ImageRef,
    PartialFinding,
    WarningEntry,
    MemoryUpdateResult,
)


class Memory(BaseModel):
    original_query: str
    confirmed: list[ConfirmedFinding] = Field(default_factory=list)
    partials: deque[PartialFinding] = Field(default_factory=lambda: deque(maxlen=2))
    warnings: list[WarningEntry] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    consolidated_summary: str = ""
    consolidated_key_facts: list[str] = Field(default_factory=list)
    iter: int = 0
    image_dir: Path | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _image_seq: int = PrivateAttr(default=0)

    def write(
        self,
        analyse_result: AnalyseResult,
        image: Image.Image,
        search_query: str,
        page_label: str | None = None,
    ) -> None:
        analyse_result.validate_branch()
        self._append_observation(analyse_result, search_query, page_label=page_label)

        if analyse_result.decision == "yes":
            resized = resize_to_pixel_count(image, 400_000)
            image_ref = self._store_image(
                resized,
                pixel_budget=400_000,
                kind="confirmed",
                page_label=page_label,
            )
            self.confirmed.append(
                ConfirmedFinding(
                    answer_text=analyse_result.answer_text or self._analysis_text(analyse_result),
                    summary=analyse_result.summary,
                    key_facts=analyse_result.key_facts,
                    image_ref=image_ref,
                    cited_statements=analyse_result.cited_statements or [],
                    found_via_query=search_query,
                )
            )
            return

        if analyse_result.decision == "partial":
            resized = resize_to_pixel_count(image, 200_000)
            image_ref = self._store_image(
                resized,
                pixel_budget=200_000,
                kind="partial",
                page_label=page_label,
            )
            self.partials.append(
                PartialFinding(
                    partial_answer=analyse_result.partial_answer or self._analysis_text(analyse_result),
                    summary=analyse_result.summary,
                    key_facts=analyse_result.key_facts,
                    image_ref=image_ref,
                    cited_statements=analyse_result.cited_statements or [],
                    remaining_gap=analyse_result.remaining_gap or "",
                    found_via_query=search_query,
                    found_at_iter=self.iter,
                )
            )
            return

        self.warnings.append(
            WarningEntry(
                query_tried=search_query,
                summary=analyse_result.summary,
                failure_type=analyse_result.failure_type or "query_too_broad",
                suggestion=analyse_result.suggestion or "",
                image_id=self._image_digest(image),
            )
        )

    def add_empty_search_warning(self, search_query: str) -> None:
        self.warnings.append(
            WarningEntry(
                query_tried=search_query,
                summary="No pages were retrieved for this query.",
                failure_type="query_too_narrow",
                suggestion="No pages were retrieved; broaden or rephrase the query.",
                image_id="retriever_empty",
            )
        )

    def context_for_decide(self) -> str:
        return (
            "Compact memory for next step:\n"
            f"{self._compact_memory_text()}"
        )

    def context_for_analyse(self) -> str:
        return (
            "Compact memory for current question:\n"
            f"{self._compact_memory_text()}"
        )

    def context_for_answer(self) -> str:
        return (
            "Compact memory:\n"
            f"{self._compact_memory_text()}\n\n"
            "Confirmed findings:\n"
            f"{self._confirmed_text(include_images=True)}\n\n"
            "Partial findings:\n"
            f"{self._partials_text(include_images=True)}\n\n"
            "Failed search attempts:\n"
            f"{self._warnings_text()}"
        )


    def append_consolidated_summary(self, summary: str, *, search_query: str | None = None) -> None:
        summary = summary.strip()
        if not summary:
            return
        if search_query:
            summary = f"[query: {search_query}] {summary}"
        if self.consolidated_summary:
            self.consolidated_summary = f"{self.consolidated_summary}\n{summary}"
        else:
            self.consolidated_summary = summary

    def update_consolidated(self, update: MemoryUpdateResult) -> None:
        update.validate_branch()
        self.consolidated_key_facts = [fact.strip() for fact in update.key_facts if fact.strip()]

    def retained_image_paths(self) -> list[Path]:
        paths: list[Path] = []
        for item in self.confirmed:
            if item.image_ref.path:
                paths.append(Path(item.image_ref.path))
        for item in self.partials:
            if item.image_ref.path:
                paths.append(Path(item.image_ref.path))
        return paths

    def visual_pixel_budget(self) -> int:
        return sum(item.image_ref.pixel_budget for item in self.confirmed) + sum(
            item.image_ref.pixel_budget for item in self.partials
        )

    def is_empty(self) -> bool:
        return not (self.confirmed or self.partials or self.warnings)

    def as_serializable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


    def _append_observation(
        self,
        analyse_result: AnalyseResult,
        search_query: str,
        *,
        page_label: str | None,
    ) -> None:
        observation: dict[str, Any] = {
            "iter": self.iter,
            "search_query": search_query,
            "page_label": page_label,
            "decision": analyse_result.decision,
            "summary": analyse_result.summary,
        }
        if analyse_result.decision in {"yes", "partial"}:
            observation["key_facts"] = analyse_result.key_facts
            observation["cited_statements"] = analyse_result.cited_statements or []
        if analyse_result.answer_text:
            observation["answer_text"] = analyse_result.answer_text
        if analyse_result.partial_answer:
            observation["partial_answer"] = analyse_result.partial_answer
        if analyse_result.remaining_gap:
            observation["remaining_gap"] = analyse_result.remaining_gap
        if analyse_result.failure_type:
            observation["failure_type"] = analyse_result.failure_type
        if analyse_result.suggestion:
            observation["suggestion"] = analyse_result.suggestion
        self.observations.append(observation)

    @staticmethod
    def _analysis_text(analyse_result: AnalyseResult) -> str:
        if analyse_result.key_facts:
            return "; ".join(analyse_result.key_facts)
        return analyse_result.summary

    def _compact_memory_text(self) -> str:
        if not self.consolidated_summary and not self.consolidated_key_facts:
            return "  None yet."
        return (
            f"  summary: {self.consolidated_summary or 'None'}\n"
            f"  key_facts: {self.consolidated_key_facts or []}"
        )

    def _confirmed_text(self, *, include_images: bool = False) -> str:
        if not self.confirmed:
            return "  None"
        lines: list[str] = []
        for idx, finding in enumerate(self.confirmed, start=1):
            image_line = ""
            if include_images:
                image_line = (
                    f"\n      image_ref: {finding.image_ref.page_label}"
                    f" ({finding.image_ref.pixel_budget} px budget)"
                )
            cited = finding.cited_statements or []
            lines.append(
                f"  [{idx}] (found via \"{finding.found_via_query}\")"
                f"{image_line}\n"
                f"      answer: {finding.answer_text}\n"
                f"      summary: {finding.summary}\n"
                f"      key_facts: {finding.key_facts}\n"
                f"      cited: {cited}"
            )
        return "\n".join(lines)

    def _partials_text(self, *, include_images: bool = False) -> str:
        if not self.partials:
            return "  None"
        lines: list[str] = []
        for finding in self.partials:
            image_line = ""
            if include_images:
                image_line = (
                    f"\n      image_ref: {finding.image_ref.page_label}"
                    f" ({finding.image_ref.pixel_budget} px budget)"
                )
            cited = finding.cited_statements or []
            lines.append(
                f"  [iter {finding.found_at_iter}] (query: \"{finding.found_via_query}\")"
                f"{image_line}\n"
                f"      partial: {finding.partial_answer}\n"
                f"      summary: {finding.summary}\n"
                f"      key_facts: {finding.key_facts}\n"
                f"      gap: {finding.remaining_gap}\n"
                f"      cited: {cited}"
            )
        return "\n".join(lines)

    def _warnings_text(self) -> str:
        if not self.warnings:
            return "  None"
        lines = []
        for warning in self.warnings:
            lines.append(
                f"  - query \"{warning.query_tried}\" -> {warning.failure_type}: "
                f"{warning.suggestion}\n"
                f"    page_summary: {warning.summary}"
            )
        return "\n".join(lines)

    def _store_image(
        self,
        image: Image.Image,
        *,
        pixel_budget: int,
        kind: str,
        page_label: str | None,
    ) -> ImageRef:
        image_id = self._image_digest(image)
        path: Path | None = None
        if self.image_dir is not None:
            self.image_dir.mkdir(parents=True, exist_ok=True)
            self._image_seq += 1
            budget_label = f"{pixel_budget // 1000}k"
            path = self.image_dir / f"{kind}_{self._image_seq:02d}_{budget_label}.png"
            image.save(path)
        return ImageRef(
            id=image_id,
            page_label=page_label or image_id,
            pixel_budget=pixel_budget,
            path=str(path) if path is not None else None,
        )

    @staticmethod
    def _image_digest(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return hashlib.sha1(buffer.getvalue()).hexdigest()[:16]
