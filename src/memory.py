from __future__ import annotations

import hashlib
import io
import json
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.image_utils import highlight_cells, resize_to_pixel_count
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
    consolidated_key_facts: list[str] = Field(default_factory=list)
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
        }
        if analyse_result.decision != "no":
            observation["key_facts"] = list(analyse_result.key_facts)
        self.observations.append(observation)

        if analyse_result.decision == "yes":
            evidence_image = retained_image or highlight_cells(image, analyse_result.useful_cells)
            resized = resize_to_pixel_count(evidence_image, 400_000)
            image_ref = self._store_image(
                resized,
                pixel_budget=400_000,
                kind="confirmed",
                page_label=page_label,
            )
            self.confirmed.append(
                ConfirmedFinding(
                    answer_text=analyse_result.answer_text
                    or " ".join(analyse_result.key_facts)
                    or analyse_result.summary,
                    image_ref=image_ref,
                    cited_statements=analyse_result.cited_statements or [],
                    found_via_query=search_query,
                    summary=analyse_result.summary,
                    key_facts=list(analyse_result.key_facts),
                    useful_cells=list(analyse_result.useful_cells),
                )
            )
            return

        if analyse_result.decision == "partial":
            evidence_image = retained_image or highlight_cells(image, analyse_result.useful_cells)
            resized = resize_to_pixel_count(evidence_image, 400_000)
            image_ref = self._store_image(
                resized,
                pixel_budget=400_000,
                kind="partial",
                page_label=page_label,
            )
            self.partials.append(
                PartialFinding(
                    partial_answer=analyse_result.partial_answer or "",
                    image_ref=image_ref,
                    cited_statements=analyse_result.cited_statements or [],
                    remaining_gap=analyse_result.remaining_gap or "",
                    found_via_query=search_query,
                    found_at_iter=self.iter,
                    summary=analyse_result.summary,
                    key_facts=list(analyse_result.key_facts),
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
                suggestion="No pages were retrieved; broaden or rephrase the query.",
                image_id="retriever_empty",
            )
        )

    def add_duplicate_search_warning(self, search_query: str, page_labels: list[str]) -> None:
        pages = ", ".join(page_labels)
        self.warnings.append(
            WarningEntry(
                query_tried=search_query,
                failure_type="query_too_narrow",
                suggestion=(
                    "The retriever returned only pages that were already analysed"
                    f"{': ' + pages if pages else ''}; use a more specific or different query."
                ),
                image_id="retriever_duplicate",
            )
        )

    def context_for_decide(self) -> str:
        return (
            "Search history:\n"
            f"{self.consolidated_summary or 'None yet.'}\n\n"
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

    def evidence_state_json(self) -> str:
        return json.dumps(self.evidence_state.model_dump(mode="json"), ensure_ascii=False, indent=2)

    def update_evidence_state(self, update: EvidenceUpdateResult) -> None:
        update.validate_branch()
        self.evidence_state = update.evidence_state

    def add_key_facts(self, key_facts: list[str]) -> None:
        seen = set(self.consolidated_key_facts)
        for fact in key_facts or []:
            text = str(fact).strip()
            if text and text not in seen:
                self.consolidated_key_facts.append(text)
                seen.add(text)

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
