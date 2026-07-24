from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


ActionType = str
SUFFICIENT_STATUS = "sufficient"


@dataclass
class QueryRecord:
    query: str
    outcome: str = "pending"
    summary: str = ""
    reason: str = ""
    evidence_refs: list[str] = field(default_factory=list)

    def as_serializable(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "outcome": self.outcome,
            "summary": self.summary,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass
class ClueNode:
    id: str
    question: str
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    status: str = "open"
    answer: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    # Each entry is a {"fact": str} dict (AGv2: facts carry no bbox).
    known_facts: list[dict] = field(default_factory=list)
    query_history: list[QueryRecord] = field(default_factory=list)
    num_attempts: int = 0

    def as_serializable(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "status": self.status,
            "answer": self.answer,
            "evidence_refs": list(self.evidence_refs),
            "known_facts": list(self.known_facts),
            "query_history": [record.as_serializable() for record in self.query_history],
            "num_attempts": self.num_attempts,
        }


@dataclass
class ClueGraph:
    nodes: dict[str, ClueNode]
    root_id: str
    active_node_id: str
    # Monotonic node-id counter. EXPAND collapse deletes nodes, so ids must never be
    # reused (len(nodes)-based ids would collide after a deletion). N0 is the root.
    next_index: int = 1

    @classmethod
    def from_root_question(cls, root_question: str) -> "ClueGraph":
        root = ClueNode(id="N0", question=root_question)
        return cls(nodes={root.id: root}, root_id=root.id, active_node_id=root.id, next_index=1)

    def active(self) -> ClueNode:
        return self.nodes[self.active_node_id]

    def root(self) -> ClueNode:
        return self.nodes[self.root_id]

    def new_node_id(self) -> str:
        node_id = f"N{self.next_index}"
        self.next_index += 1
        return node_id

    def add_child(self, parent_id: str, question: str) -> str:
        child_id = self.new_node_id()
        child = ClueNode(
            id=child_id,
            question=question,
            parent_id=parent_id,
        )
        self.nodes[child_id] = child
        self.nodes[parent_id].children.append(child_id)
        return child_id

    def set_active(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        self.active_node_id = node_id

    def as_serializable(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "active_node_id": self.active_node_id,
            "nodes": {
                node_id: node.as_serializable()
                for node_id, node in sorted(self.nodes.items())
            },
        }


@dataclass
class ObservationImage:
    image_id: str
    page_label: str
    image: Any
    path: str | None = None
    display_image: Any | None = None
    source_image_id: str | None = None
    crop_box: list[float] | None = None
    cells: list[str] = field(default_factory=list)
    summary: str = ""

    def prompt_image(self) -> Any:
        return self.display_image if self.display_image is not None else self.image

    def evidence_ref(self) -> str:
        if self.crop_box is not None or self.cells:
            return self.image_id
        return self.page_label or self.image_id

    def as_serializable(self) -> dict[str, Any]:
        size = getattr(self.image, "size", None)
        display_size = getattr(self.prompt_image(), "size", None)
        return {
            "image_id": self.image_id,
            "page_label": self.page_label,
            "path": self.path,
            "source_image_id": self.source_image_id,
            "crop_box": list(self.crop_box) if self.crop_box is not None else None,
            "cells": list(self.cells),
            "summary": self.summary,
            "size": list(size) if size else None,
            "display_size": list(display_size) if display_size else None,
        }


@dataclass
class VisualObservation:
    query: str
    images: list[ObservationImage] = field(default_factory=list)
    kind: str = "search"
    message: str = ""

    def prompt_images(self) -> list[Any]:
        return [image.prompt_image() for image in self.images]

    def evidence_refs(self) -> list[str]:
        refs: list[str] = []
        seen: set[str] = set()
        for image in self.images:
            ref = image.evidence_ref()
            if ref and ref not in seen:
                refs.append(ref)
                seen.add(ref)
        return refs

    def summary(self) -> str:
        parts: list[str] = []
        if self.message:
            parts.append(self.message)
        for image in self.images:
            label = image.page_label or image.image_id
            if image.summary:
                parts.append(f"{image.image_id} ({label}): {image.summary}")
            else:
                parts.append(f"{image.image_id} ({label})")
        if parts:
            return " | ".join(parts)
        return "No visual candidates were returned."

    def as_serializable(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "query": self.query,
            "message": self.message,
            "images": [image.as_serializable() for image in self.images],
        }


def is_sufficient(node: ClueNode) -> bool:
    return node.status in {SUFFICIENT_STATUS, "established"}


def can_update_from_graph(node: ClueNode, graph: ClueGraph) -> bool:
    if can_update_from_active_node_evidence(node, graph):
        return True
    return can_update_from_parent_context(node, graph)


def can_update_from_active_node_evidence(node: ClueNode, graph: ClueGraph) -> bool:
    if node.known_facts:
        return True
    return any(is_sufficient(graph.nodes[child_id]) for child_id in node.children)


def can_update_from_parent_context(node: ClueNode, graph: ClueGraph) -> bool:
    if node.parent_id is not None:
        parent = graph.nodes[node.parent_id]
        if parent.known_facts:
            return True
        for sibling_id in parent.children:
            sibling = graph.nodes[sibling_id]
            if is_sufficient(sibling):
                return True

    return False


def compact_query_history(node: ClueNode, max_items: int = 2) -> list[dict[str, str]]:
    bad_outcomes = {"failure", "weak", "irrelevant", "unreadable"}
    failed = [record for record in node.query_history if record.outcome in bad_outcomes]
    useful = [
        record
        for record in node.query_history
        if record.outcome in {"partial", "success", "led_to_expand", "led_to_establish"}
    ]
    selected = failed[-max_items:] + useful[-1:]
    return [
        {
            "query": record.query,
            "outcome": record.outcome,
            "summary": record.summary,
            "reason": record.reason,
        }
        for record in selected
    ]


def sufficient_child_answers(node: ClueNode, graph: ClueGraph) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for child_id in node.children:
        child = graph.nodes[child_id]
        if not is_sufficient(child):
            continue
        rows.append(
            {
                "id": child.id,
                "question": child.question,
                "answer": child.answer,
                "known_facts": list(child.known_facts),
                "evidence_refs": list(child.evidence_refs),
            }
        )
    return rows


def update_last_query_record(
    node: ClueNode,
    decision: Any,
    observation_summary: str = "",
) -> None:
    if not node.query_history:
        return

    record = node.query_history[-1]
    if record.outcome != "pending":
        return

    decision_type = getattr(decision, "type")

    if decision_type == "REJECT":
        record.outcome = "failure"
        record.summary = (
            getattr(decision, "summary", None)
            or observation_summary
            or "The retrieved observation did not produce reliable evidence."
        )
        record.reason = (
            getattr(decision, "reason", None)
            or "The next SEARCH action should use a different retrieval direction."
        )
        record.evidence_refs = []

    elif decision_type == "EXPAND":
        known_facts = [_fact_label(_f) for _f in (getattr(decision, "supporting_facts", []) or [])]
        evidence_refs = list(getattr(decision, "evidence_refs", []) or [])
        record.outcome = "partial"
        record.summary = (
            getattr(decision, "summary", None)
            or observation_summary
            or "; ".join(known_facts)
            or getattr(decision, "answer", None)
            or ""
        )
        record.reason = (
            getattr(decision, "reason", None)
            or "The query answered one subquestion but left a remaining subquestion."
        )
        record.evidence_refs = evidence_refs

    elif decision_type == "ACCEPT":
        evidence_refs = list(getattr(decision, "evidence_refs", []) or [])
        record.outcome = "success"
        record.summary = (
            getattr(decision, "summary", None)
            or observation_summary
            or getattr(decision, "answer", None)
            or ""
        )
        record.reason = (
            getattr(decision, "reason", None)
            or "The query found enough evidence to make the active node sufficient."
        )
        record.evidence_refs = evidence_refs


def apply_graph_decision(
    graph: ClueGraph,
    node_id: str,
    decision: Any,
    observation_summary: str = "",
) -> None:
    node = graph.nodes[node_id]
    decision_type = getattr(decision, "type")

    if decision_type == "ACCEPT":
        if not getattr(decision, "answer", None):
            raise ValueError("ACCEPT requires answer")
        node.status = SUFFICIENT_STATUS
        node.answer = str(decision.answer)
        node.evidence_refs.extend(_compact_new(node.evidence_refs, decision.evidence_refs))
        # 1b (2026-06-13): root ACCEPT OVERWRITES its facts with the final synthesis;
        # non-root nodes keep accumulating. Empty supporting_facts falls back to [answer].
        # overwrite facts on the ROOT or any EXPANDed (>=2 children) synthesis node;
        # leaf / non-expanded non-root nodes still accumulate (extend).
        if node.id == graph.root_id or len(node.children) >= 2:
            node.known_facts = [_fact_entry(_f) for _f in (decision.supporting_facts or [])] or [{"fact": str(decision.answer)}]
        else:
            node.known_facts.extend(_compact_new(node.known_facts, [_fact_entry(_f) for _f in (decision.supporting_facts or [])]))
        update_last_query_record(node, decision, observation_summary)
        graph.set_active(node.parent_id or node.id)
        return

    if decision_type == "REJECT":
        update_last_query_record(node, decision, observation_summary)
        graph.set_active(node.id)
        return

    if decision_type == "EXPAND":
        answered_subquestion = getattr(decision, "answered_subquestion", None)
        answer = getattr(decision, "answer", None)
        remaining_subquestion = getattr(decision, "remaining_subquestion", None)
        supporting_facts = [_fact_entry(_f) for _f in (getattr(decision, "supporting_facts", []) or [])]
        if not answered_subquestion:
            raise ValueError("EXPAND requires answered_subquestion")
        if not answer:
            raise ValueError("EXPAND requires answer")
        if not remaining_subquestion:
            raise ValueError("EXPAND requires remaining_subquestion")
        if not supporting_facts:
            supporting_facts = [{"fact": str(answer)}]
        # 1c (2026-06-13): EXPAND no longer writes facts to the parent; they live only on the
        # answered_child below. evidence_refs still propagate to parent for provenance.
        node.evidence_refs.extend(_compact_new(node.evidence_refs, decision.evidence_refs))
        # Collapse on re-EXPAND (2026-06-19): if this node already expanded before (it has
        # children), absorb their known_facts + evidence_refs into this node and drop the
        # old subtrees, so the node keeps only the fresh [answered(sufficient), remaining(open)]
        # pair created below. Keeps graph_state shallow without silently losing evidence.
        # next_index is monotonic, so the dropped node ids are never reused.
        if node.children:
            for _old_child_id in list(node.children):
                _drop_subtree_absorbing_facts(graph, node, _old_child_id)
            node.children = []
        answered_child_id = graph.add_child(
            parent_id=node.id,
            question=str(answered_subquestion),
        )
        answered_child = graph.nodes[answered_child_id]
        answered_child.status = SUFFICIENT_STATUS
        answered_child.answer = str(answer)
        answered_child.evidence_refs.extend(
            _compact_new(answered_child.evidence_refs, decision.evidence_refs)
        )
        answered_child.known_facts.extend(
            _compact_new(answered_child.known_facts, supporting_facts)
        )
        remaining_child_id = graph.add_child(
            parent_id=node.id,
            question=str(remaining_subquestion),
        )
        update_last_query_record(node, decision, observation_summary)
        graph.set_active(remaining_child_id)
        return

    raise ValueError(f"Unknown graph decision type: {decision_type}")


def _drop_subtree_absorbing_facts(graph: "ClueGraph", target: "ClueNode", node_id: str) -> None:
    """Roll node_id's (and its descendants') known_facts + evidence_refs into ``target``,
    then remove the node and its whole subtree from the graph. Used by EXPAND collapse so a
    re-expanding node keeps only its newest [answered, remaining] pair without silently
    losing accumulated evidence (option ii). Node ids are not reused (monotonic next_index)."""
    node = graph.nodes.get(node_id)
    if node is None:
        return
    for child_id in list(node.children):
        _drop_subtree_absorbing_facts(graph, target, child_id)
    if node.id != target.id:
        target.known_facts.extend(_compact_new(target.known_facts, node.known_facts))
        target.evidence_refs.extend(_compact_new(target.evidence_refs, node.evidence_refs))
        graph.nodes.pop(node_id, None)


def _node_order_key(node_id: str):
    try:
        return (0, int(str(node_id)[1:]))
    except (TypeError, ValueError):
        return (1, str(node_id))


def format_graph_state_for_prompt(graph: ClueGraph) -> str:
    active = graph.active()
    rows = ["Graph nodes:"]
    for node_id in sorted(graph.nodes, key=_node_order_key):
        node = graph.nodes[node_id]
        tags = [f"[{node.id}]"]
        if node.id == graph.root_id:
            tags.append("root")
        tags.append(f"({node.status})")
        if node.id == graph.active_node_id:
            tags.append("<<ACTIVE>>")
        children = ", ".join(node.children)
        rows.append(" ".join(tags) + f" children=[{children}]")
        rows.append(f"     question: {node.question}")
        if node.answer:
            rows.append(f"     answer:   {node.answer}")
        if node.known_facts or node.id == graph.active_node_id:
            rows.append(f"     facts:    {_format_facts(node.known_facts)}")
    rows.append(f"Active: [{active.id}]")
    query_history = compact_query_history(active)
    if query_history:
        rows.append(f"Recent queries: {query_history}")
    return "\n".join(rows)

def format_observation_for_prompt(observation: VisualObservation | None) -> str:
    if observation is None:
        return "Current visual observation: None."

    rows = [
        "Current visual observation:",
        f"  kind: {observation.kind}",
        f"  search_query: {observation.query}",
    ]
    if not observation.images:
        rows.append("  images: []")
        return "\n".join(rows)

    rows.append("  images:")
    for image in observation.images:
        rows.append(f"  - image_id: {image.image_id}; page_label: {image.page_label}")
        rows.append("    <image>")
    return "\n".join(rows)


def _compact_new(existing: list, values: list | None) -> list:
    """Values from `values` not already in `existing`, deduped by fact-text. Preserves item type
    so it works for both str evidence_refs and {"fact","bbox_2d"} dicts."""
    seen = {_fact_label(item).strip().lower() for item in existing}
    out: list = []
    for value in values or []:
        key = _fact_label(value).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _format_facts(facts: list) -> str:
    """Render known_facts as a JSON list of fact strings, e.g. ["X is 12%", "Y in 2021"].
    bbox_2d is intentionally omitted from graph_state (old-protocol-validated design): the
    source images for past rounds are no longer in context, so coordinates would be noise.
    The model still EMITS bbox_2d in update_graph (grounding discipline at write time) and
    bbox is still stored on the node via _fact_entry (provenance / reward) — only the
    rendered prompt drops it. Facts superseded by a later zoom re-read are also skipped
    (storage keeps them; the prompt must not show stale readings next to refined ones)."""
    visible = [f for f in (facts or []) if not (isinstance(f, dict) and f.get("superseded"))]
    return json.dumps([_fact_label(f) for f in visible], ensure_ascii=False)


def _fact_entry(f) -> dict:
    """Normalize a fact given as str / dict / pydantic model into {"fact": str} plus an
    optional "bbox_2d" (AGv2.1 grounded facts; page-pixel frame)."""
    box = None
    if isinstance(f, dict):
        fact = str(f.get("fact") or f.get("label") or f.get("text") or "").strip()
        box = f.get("bbox_2d")
    elif isinstance(f, str):
        fact = f.strip()
    else:
        fact = str(getattr(f, "fact", "") or "").strip()
        box = getattr(f, "bbox_2d", None)
    entry: dict = {"fact": fact}
    try:
        if box and len(box) == 4:
            entry["bbox_2d"] = [float(v) for v in box]
    except (TypeError, ValueError):
        pass
    return entry


def _fact_label(f):
    """Extract the text label from a SupportingFact / dict / str (bbox-facts refactor)."""
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return str(f.get("fact") or f.get("label") or f.get("text") or "")
    return str(getattr(f, "fact", "") or "")
