from __future__ import annotations

import hashlib
import io
import json
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.image_utils import crop_grid_cells_bbox, resize_to_pixel_count
import os as _os
_EVAL_PX = int(_os.environ.get("ACTIVE_GRAPH_EVAL_PIXELS", "400000"))
_EVAL_UP = _EVAL_PX <= 600000  # native passthrough (no up/down-scale) when budget large

from src.schemas import (
    AnalyseResult,
    ConfirmedFinding,
    EvidenceState,
    EvidenceUpdateResult,
    ImageRef,
    PartialFinding,
    WarningEntry,
)


class Memory(BaseModel):
    original_query: str
    confirmed: list[ConfirmedFinding] = Field(default_factory=list)
    partials: deque[PartialFinding] = Field(default_factory=lambda: deque(maxlen=2))
    warnings: list[WarningEntry] = Field(default_factory=list)
    consolidated_summary: str = ""
    consolidated_summaries: list[str] = Field(default_factory=list)
    evidence_state: EvidenceState = Field(default_factory=EvidenceState)
    observations: list[dict[str, Any]] = Field(default_factory=list)
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
        retained_image: Image.Image | None = None,
    ) -> None:
        analyse_result.validate_branch()

        observation = {
            "iter": self.iter,
            "search_query": search_query,
            "page_label": page_label,
            "decision": analyse_result.decision,
            "useful_cells": list(analyse_result.useful_cells),
            "summary": analyse_result.summary,
            "observed_evidence": analyse_result.observed_evidence,
        }
        self.observations.append(observation)

        if analyse_result.decision == "yes":
            if retained_image is not None:
                evidence_image = retained_image
                resized = resize_to_pixel_count(evidence_image, _EVAL_PX, allow_upscale=_EVAL_UP)
                image_ref = self._store_image(
                    resized,
                    pixel_budget=_EVAL_PX,
                    kind="confirmed",
                    page_label=page_label,
                )
            elif analyse_result.useful_cells:
                evidence_image = crop_grid_cells_bbox(image, analyse_result.useful_cells)
                resized = resize_to_pixel_count(evidence_image, _EVAL_PX, allow_upscale=_EVAL_UP)
                image_ref = self._store_image(
                    resized,
                    pixel_budget=_EVAL_PX,
                    kind="confirmed",
                    page_label=page_label,
                )
            else:
                resized = resize_to_pixel_count(image, _EVAL_PX, allow_upscale=_EVAL_UP)
                image_ref = self._store_image(
                    resized,
                    pixel_budget=_EVAL_PX,
                    kind="confirmed",
                    page_label=page_label,
                )
            self.confirmed.append(
                ConfirmedFinding(
                    answer_text=analyse_result.answer_text
                    or analyse_result.observed_evidence
                    or analyse_result.summary,
                    image_ref=image_ref,
                    cited_statements=analyse_result.cited_statements or [],
                    found_via_query=search_query,
                    summary=analyse_result.summary,
                    useful_cells=list(analyse_result.useful_cells),
                )
            )
            return

        if analyse_result.decision == "partial":
            if retained_image is not None:
                evidence_image = retained_image
                resized = resize_to_pixel_count(evidence_image, _EVAL_PX, allow_upscale=_EVAL_UP)
                image_ref = self._store_image(
                    resized,
                    pixel_budget=_EVAL_PX,
                    kind="partial",
                    page_label=page_label,
                )
            elif analyse_result.useful_cells:
                evidence_image = crop_grid_cells_bbox(image, analyse_result.useful_cells)
                resized = resize_to_pixel_count(evidence_image, _EVAL_PX, allow_upscale=_EVAL_UP)
                image_ref = self._store_image(
                    resized,
                    pixel_budget=_EVAL_PX,
                    kind="partial",
                    page_label=page_label,
                )
            else:
                resized = resize_to_pixel_count(image, _EVAL_PX, allow_upscale=_EVAL_UP)
                image_ref = self._store_image(
                    resized,
                    pixel_budget=_EVAL_PX,
                    kind="partial",
                    page_label=page_label,
                )
            self.partials.append(
                PartialFinding(
                    partial_answer=analyse_result.partial_answer or analyse_result.observed_evidence or "",
                    image_ref=image_ref,
                    cited_statements=analyse_result.cited_statements or [],
                    found_via_query=search_query,
                    found_at_iter=self.iter,
                    summary=analyse_result.summary,
                    useful_cells=list(analyse_result.useful_cells),
                )
            )
            return

        self.warnings.append(
            WarningEntry(
                query_tried=search_query,
                failure_type=analyse_result.failure_type or "query_too_broad",
                suggestion=analyse_result.suggestion or "",
                image_id=self._image_digest(image),
                summary=analyse_result.summary,
            )
        )

    def add_empty_search_warning(self, search_query: str) -> None:
        self.warnings.append(
            WarningEntry(
                query_tried=search_query,
                failure_type="query_too_narrow",
                suggestion="reframe",
                image_id="retriever_empty",
            )
        )

    def add_duplicate_search_warning(self, search_query: str, page_labels: list[str]) -> None:
        self.warnings.append(
            WarningEntry(
                query_tried=search_query,
                failure_type="query_too_narrow",
                suggestion="reframe",
                image_id="retriever_duplicate",
            )
        )

    def context_for_decide(self) -> str:
        return (
            "Successful search history (yes/partial):\n"
            f"{self.successful_search_log()}\n\n"
            "Failed search history (do not repeat these queries):\n"
            f"{self.failed_search_log()}\n\n"
            "Current evidence_state:\n"
            f"{self.evidence_state_json()}"
        )

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

    def append_consolidated_summary(self, summary: str, *, search_query: str | None = None) -> None:
        summary = str(summary or "").strip()
        if not summary:
            return
        if search_query:
            summary = f"[query: {search_query}]: {summary}"
        self.consolidated_summaries.append(summary)
        self.consolidated_summary = "\n".join(self.consolidated_summaries)

    def failed_search_log(self) -> str:
        if not self.warnings:
            return "None yet."
        rows: list[str] = []
        for warning in self.warnings[-8:]:
            detail = warning.summary or "Only repeated pages found; try a different query."
            rows.append(f"[query: {warning.query_tried}]: {detail}")
        return "\n".join(rows)

    def successful_search_log(self) -> str:
        return self.consolidated_summary or "None yet."

    def search_history_log(self) -> str:
        rows: list[str] = []
        successful = self.successful_search_log()
        if successful != "None yet.":
            rows.append(successful)
        failed = self.failed_search_log()
        if failed != "None yet.":
            rows.append(failed)
        return "\n".join(rows) if rows else "None yet."

    def evidence_state_for_prompt(self) -> dict[str, Any]:
        return {
            "answer_relevant_facts": list(self.evidence_state.answer_relevant_facts),
            "missing_requirements": list(self.evidence_state.missing_requirements),
        }

    def evidence_state_json(self) -> str:
        return json.dumps(self.evidence_state_for_prompt(), ensure_ascii=False, indent=2)

    def update_evidence_state(self, update: EvidenceUpdateResult) -> None:
        update.validate_branch()
        self.evidence_state = update.evidence_state

    def append_observed_evidence(self, observed_evidence: str) -> None:
        text = str(observed_evidence or "").strip()
        if not text:
            return
        facts = list(self.evidence_state.answer_relevant_facts)
        if text.lower() not in {item.lower() for item in facts}:
            facts.append(text)
        self.evidence_state.answer_relevant_facts = facts
        self.evidence_state.normalize()

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
