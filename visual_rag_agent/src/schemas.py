from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


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
    useful_cells: list[str] = Field(default_factory=list)


class PartialFinding(BaseModel):
    partial_answer: str
    image_ref: ImageRef
    cited_statements: list[str] = Field(default_factory=list)
    found_via_query: str
    found_at_iter: int
    summary: str = ""
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
    observed_evidence: str = ""
    # Analyse no longer emits judge in text; evidence_update assigns the page-level judge.
    # Keep aliases for compatibility with older trajectories/checkpoints.
    decision: Literal["yes", "partial", "no"] = Field(
        default="partial", validation_alias=AliasChoices("decision", "judge")
    )
    answer_text: str | None = None
    cited_statements: list[str] | None = None
    partial_answer: str | None = None
    failure_type: FailureType | None = None
    suggestion: str | None = None

    @property
    def judge(self) -> Literal["yes", "partial", "no"]:
        return self.decision

    def validate_branch(self) -> None:
        if self.decision == "yes":
            if not self.observed_evidence and self.summary:
                self.observed_evidence = self.summary
            if not self.answer_text and self.observed_evidence:
                self.answer_text = self.observed_evidence
            if not self.answer_text and self.summary:
                self.answer_text = self.summary
            return

        if self.decision == "partial":
            if not self.observed_evidence and self.summary:
                self.observed_evidence = self.summary
            if not self.partial_answer and self.observed_evidence:
                self.partial_answer = self.observed_evidence
            if not self.partial_answer and self.summary:
                self.partial_answer = self.summary
            return

        if not self.failure_type:
            self.failure_type = "wrong_doc_section"
        if not self.suggestion:
            self.suggestion = "Search for a page with evidence that directly addresses the question."
        self.useful_cells = []
        self.observed_evidence = ""


class EvidenceState(BaseModel):
    answer_relevant_facts: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(
        default_factory=lambda: ["No evidence has been gathered yet."]
    )
    observed_evidence: str = ""

    def normalize(self) -> None:
        self.answer_relevant_facts = compact_unique(self.answer_relevant_facts)
        self.missing_requirements = compact_unique(self.missing_requirements)
        if not self.observed_evidence.strip() and self.answer_relevant_facts:
            self.observed_evidence = "\n".join(self.answer_relevant_facts)
        else:
            self.observed_evidence = self.observed_evidence.strip()


class EvidenceUpdateResult(BaseModel):
    evidence_state: EvidenceState
    judge: Literal["yes", "partial", "no"] = "partial"

    def validate_branch(self) -> None:
        self.evidence_state.normalize()


def compact_unique(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = " ".join(str(value or "").strip().split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(text)
    return rows


class DecideResult(BaseModel):
    think: str
    action: Literal["search", "answer"]
    search_query: str | None = None
    content: str | None = None
    answer: str | None = Field(
        default=None,
        validation_alias=AliasChoices("answer", "value"),
    )

    @model_validator(mode="after")
    def fill_content_aliases(self) -> "DecideResult":
        if self.content is None and self.search_query is not None:
            self.content = self.search_query
        if self.content is None and self.answer is not None:
            self.content = self.answer
        if self.search_query is None and self.action == "search" and self.content is not None:
            self.search_query = self.content
        if self.answer is None and self.action == "answer" and self.content is not None:
            self.answer = self.content
        return self

    def validate_branch(self) -> None:
        if self.action == "search" and not self.search_query:
            raise ValueError("search_query required when action is search")
        if self.action == "answer" and not self.content:
            raise ValueError("answer content required when action is answer")


class PolicyActionResult(BaseModel):
    think: str = ""
    type: Literal["SEARCH", "CROP", "UPDATE_GRAPH", "ANSWER"]
    query: str | None = None
    search_query: str | None = None
    image_id: str | None = None
    box: list[float] | None = None
    cells: list[str] = Field(default_factory=list)
    reason: str | None = None
    graph_update: dict[str, Any] | None = None
    answer: str | None = None

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: object) -> str:
        return str(value).upper()

    @model_validator(mode="after")
    def fill_search_query_aliases(self) -> "PolicyActionResult":
        if self.query is None and self.search_query is not None:
            self.query = self.search_query
        if self.search_query is None and self.query is not None:
            self.search_query = self.query
        return self

    def validate_branch(self, valid_actions: list[str] | None = None) -> None:
        if valid_actions is not None and self.type not in valid_actions:
            raise ValueError(f"{self.type} is not in valid actions: {valid_actions}")
        if self.type == "SEARCH" and not self.query:
            raise ValueError("SEARCH requires query")
        if self.type == "CROP" and self.box is None and not self.cells:
            raise ValueError("CROP requires box or cells")
        if self.type == "ANSWER" and not (self.answer or "").strip():
            raise ValueError("ANSWER requires answer")
        if self.type == "CROP" and self.box is not None:
            if len(self.box) != 4:
                raise ValueError("CROP box requires four values")
            if any(value < 0.0 for value in self.box):
                raise ValueError("CROP box values must be non-negative")
            if self.box[2] <= self.box[0] or self.box[3] <= self.box[1]:
                raise ValueError("CROP box must satisfy x1 > x0 and y1 > y0")


class SupportingFact(BaseModel):
    """A grounded fact: text + the image region (Qwen2.5-VL bbox_2d) that supports it."""
    fact: str
    bbox_2d: list[float]

    @field_validator("bbox_2d")
    @classmethod
    def _check_bbox(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("bbox_2d requires four values [x1,y1,x2,y2]")
        if any(float(c) < 0 for c in v):
            raise ValueError("bbox_2d values must be non-negative")
        if float(v[2]) <= float(v[0]) or float(v[3]) <= float(v[1]):
            raise ValueError("bbox_2d must satisfy x2 > x1 and y2 > y1")
        return [float(c) for c in v]


class GraphDecisionResult(BaseModel):
    type: Literal["ACCEPT", "REJECT", "EXPAND"]

    answer: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    supporting_facts: list[SupportingFact] = Field(default_factory=list)

    summary: str | None = Field(
        default=None,
        validation_alias=AliasChoices("summary", "query_failure_summary"),
    )
    reason: str | None = Field(
        default=None,
        validation_alias=AliasChoices("reason", "query_failure_reason"),
    )

    answered_subquestion: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "answered_subquestion",
            "subquestion_answered",
            "answered_question",
        ),
    )
    remaining_subquestion: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "remaining_subquestion",
            "remaining_question",
            "subquestion_remaining",
            "child_question",
        ),
    )

    known_facts: list[str] = Field(default_factory=list)

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: object) -> str:
        return str(value).upper()

    @model_validator(mode="before")
    @classmethod
    def normalize_model_aliases(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        decision_type = str(out.get("type", "")).upper()
        if decision_type == "ACCEPT" and "answer" not in out and "value" in out:
            out["answer"] = out["value"]
        if decision_type == "EXPAND" and "answer" not in out:
            for key in ("answered_answer", "subquestion_answer", "answer_to_subquestion"):
                if key in out:
                    out["answer"] = out[key]
                    break
            if "answer" not in out and "value" in out:
                out["answer"] = out["value"]
        if decision_type == "EXPAND" and "supporting_facts" not in out:
            for key in ("answered_facts", "facts", "known_facts"):
                if key in out:
                    out["supporting_facts"] = out[key]
                    break
        sf = out.get("supporting_facts")
        if isinstance(sf, list):
            coerced = []
            for f in sf:
                if isinstance(f, str):
                    coerced.append({"fact": f})  # no bbox_2d -> required-field error
                elif isinstance(f, dict):
                    g = dict(f)
                    if "fact" not in g:
                        for k in ("label", "text", "statement", "answer"):
                            if k in g:
                                g["fact"] = g[k]; break
                    if "bbox_2d" not in g:
                        for k in ("bbox", "box", "bbox2d", "bounding_box"):
                            if k in g:
                                g["bbox_2d"] = g[k]; break
                    coerced.append(g)
                else:
                    coerced.append(f)
            out["supporting_facts"] = coerced
        if "answered_subquestion" not in out:
            for key in ("subquestion_answered", "answered_question"):
                if key in out:
                    out["answered_subquestion"] = out[key]
                    break
        if "remaining_subquestion" not in out:
            for key in ("remaining_question", "subquestion_remaining", "child_question"):
                if key in out:
                    out["remaining_subquestion"] = out[key]
                    break
        return out

    @model_validator(mode="after")
    def validate_branch(self) -> "GraphDecisionResult":
        if self.type == "ACCEPT":
            if not self.answer:
                raise ValueError("ACCEPT requires answer")
            if not self.summary:
                self.summary = self.answer
            return self

        if self.type == "REJECT":
            if self.known_facts:
                raise ValueError("REJECT must not write known_facts")
            if self.supporting_facts:
                raise ValueError("REJECT must not write supporting_facts")
            if self.answer:
                raise ValueError("REJECT must not write answer")
            if not self.summary:
                raise ValueError("REJECT requires summary")
            if not self.reason:
                raise ValueError("REJECT requires reason")
            return self

        if self.type == "EXPAND":
            if not self.answered_subquestion:
                raise ValueError("EXPAND requires answered_subquestion")
            if not self.answer:
                raise ValueError("EXPAND requires answer")
            if not self.remaining_subquestion:
                raise ValueError("EXPAND requires remaining_subquestion")
            if not self.summary:
                self.summary = self.answer
            return self

        return self


class ImageSummaryResult(BaseModel):
    summary: str
    visible_text: list[str] = Field(default_factory=list)

    def validate_branch(self) -> None:
        if not self.summary.strip():
            raise ValueError("summary is required")
