from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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
    answer_text: str
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    image_ref: ImageRef
    cited_statements: list[str] = Field(default_factory=list)
    found_via_query: str


class PartialFinding(BaseModel):
    partial_answer: str
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    image_ref: ImageRef
    cited_statements: list[str] = Field(default_factory=list)
    remaining_gap: str
    found_via_query: str
    found_at_iter: int


class WarningEntry(BaseModel):
    query_tried: str
    summary: str
    failure_type: FailureType
    suggestion: str
    image_id: str


class AnalyseResult(BaseModel):
    think: str
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    judge: Literal["yes", "partial", "no"]
    answer_text: str | None = None
    cited_statements: list[str] | None = None
    partial_answer: str | None = None
    remaining_gap: str | None = None
    failure_type: FailureType | None = None
    suggestion: str | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_decision_key(cls, data: Any) -> Any:
        if isinstance(data, dict) and "judge" not in data and "decision" in data:
            data = {**data, "judge": data["decision"]}
        return data

    @property
    def decision(self) -> Literal["yes", "partial", "no"]:
        return self.judge

    def validate_branch(self) -> None:
        if not self.summary.strip():
            raise ValueError("summary required for all analyse judgments")
        if not self.think.strip():
            raise ValueError("think required for all analyse judgments")


class MemoryUpdateResult(BaseModel):
    key_facts: list[str] = Field(default_factory=list)

    def validate_branch(self) -> None:
        return None


class DecideResult(BaseModel):
    think: str
    action: Literal["search", "answer"]
    content: str | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_search_query_key(cls, data: Any) -> Any:
        if isinstance(data, dict) and "content" not in data and "search_query" in data:
            data = {**data, "content": data["search_query"]}
        return data

    @property
    def search_query(self) -> str | None:
        if self.action == "search":
            return self.content
        return None

    def validate_branch(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("content required for both search and answer decisions")


class ImageSummaryResult(BaseModel):
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    visible_text: list[str] = Field(default_factory=list)

    def validate_branch(self) -> None:
        if not self.summary.strip():
            raise ValueError("summary is required")
