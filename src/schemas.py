from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator


FailureType = Literal[
    "wrong_domain",
    "wrong_doc_section",
    "query_too_broad",
    "query_too_narrow",
]


class ImageRef(BaseModel):
    id: str
    page_label: str
    pixel_budget: int
    path: str | None = None


class ConfirmedFinding(BaseModel):
    answer_text: str = ""
    image_ref: ImageRef
    cited_statements: list[str] = Field(default_factory=list)
    found_via_query: str
    summary: str = ""
    key_facts: list[str] = Field(default_factory=list)
    useful_cells: list[str] = Field(default_factory=list)


class PartialFinding(BaseModel):
    partial_answer: str
    image_ref: ImageRef
    cited_statements: list[str] = Field(default_factory=list)
    remaining_gap: str
    found_via_query: str
    found_at_iter: int
    summary: str = ""
    key_facts: list[str] = Field(default_factory=list)
    useful_cells: list[str] = Field(default_factory=list)


class WarningEntry(BaseModel):
    query_tried: str
    failure_type: FailureType
    suggestion: str
    image_id: str
    summary: str = ""


class AnalyseResult(BaseModel):
    think: str
    useful_cells: list[str] = Field(default_factory=list)
    summary: str = ""
    # Backward compatible fields for old trajectories. The new pipeline keeps
    # analyse focused on think/summary/useful_cells/judge and moves evidence
    # consolidation to EvidenceUpdateResult.
    key_facts: list[str] = Field(default_factory=list)
    decision: Literal["yes", "partial", "no"] = Field(
        validation_alias=AliasChoices("decision", "judge")
    )
    answer_text: str | None = None
    cited_statements: list[str] | None = None
    partial_answer: str | None = None
    remaining_gap: str | None = None
    failure_type: FailureType | None = None
    suggestion: str | None = None

    @property
    def judge(self) -> Literal["yes", "partial", "no"]:
        return self.decision

    def validate_branch(self) -> None:
        if self.decision == "yes":
            if not self.answer_text and self.key_facts:
                self.answer_text = " ".join(self.key_facts)
            if not self.answer_text and self.summary:
                self.answer_text = self.summary
            return

        if self.decision == "partial":
            if not self.partial_answer and self.key_facts:
                self.partial_answer = " ".join(self.key_facts)
            if not self.partial_answer and self.summary:
                self.partial_answer = self.summary
            if not self.remaining_gap:
                self.remaining_gap = "More evidence is needed."
            return

        if not self.failure_type:
            self.failure_type = "wrong_doc_section"
        if not self.suggestion:
            self.suggestion = "Search for a page with evidence that directly addresses the question."
        self.useful_cells = []
        self.key_facts = []


class EvidenceState(BaseModel):
    observed_evidence: str = ""
    remaining_gap: str | None = "No evidence has been gathered yet."


class EvidenceUpdateResult(BaseModel):
    evidence_state: EvidenceState

    def validate_branch(self) -> None:
        observed = self.evidence_state.observed_evidence.strip()
        gap = self.evidence_state.remaining_gap
        if not observed and gap is None:
            raise ValueError("observed_evidence is required when remaining_gap is null")
        if isinstance(gap, str):
            self.evidence_state.remaining_gap = gap.strip() or None


class DecideResult(BaseModel):
    think: str
    action: Literal["search", "answer"]
    search_query: str | None = None
    content: str | None = None

    @model_validator(mode="after")
    def fill_content_aliases(self) -> "DecideResult":
        if self.content is None and self.search_query is not None:
            self.content = self.search_query
        if self.search_query is None and self.action == "search" and self.content is not None:
            self.search_query = self.content
        return self

    def validate_branch(self) -> None:
        if self.action == "search" and not self.search_query:
            raise ValueError("search_query required when action is search")


class ImageSummaryResult(BaseModel):
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    visible_text: list[str] = Field(default_factory=list)

    def validate_branch(self) -> None:
        if not self.summary.strip():
            raise ValueError("summary is required")
