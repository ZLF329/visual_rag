from __future__ import annotations

"""new memory M = (T, P, E) — DESIGN.md §2.

T  question tree: N0 (fixed original question) + ordered sub-questions, all children of N0.
P  page nodes: one per real page seen (alias P1, P2, ...); each owns its facts (stored as P1.F1,
   shown and referenced as F1 inside the page's block).
E  question -> page references (QNode.pages); page -> question links are derived.

Every edit goes through apply_edit(): the turn's page edit (desc, facts of the page in view) and then
its tree edit (DESIGN.md §11.23: a resolve / expand drawn from the page memory that attaches exactly the
pages it lists) are validated and applied on a clone, so an invalid turn leaves the memory untouched.
Rendering never shows boxes or real page labels.
"""

import json
from dataclasses import dataclass, field, replace
from typing import Any

from PIL import Image

from src.protocol import (
    AddFact,
    Box,
    EditError,
    Expand,
    Frame,
    PageEdit,
    Resolve,
    unit_to_region,
)

ROOT_ID = "N0"
OPEN, RESOLVED = "open", "resolved"


@dataclass
class Fact:
    id: str
    text: str
    bbox: Box             # original page pixels
    src: str              # "page" | "zoom": the view it was last read from
    revision: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "bbox": [round(v, 1) for v in self.bbox], "src": self.src,
                "revision": self.revision, "history": list(self.history)}


@dataclass
class Visit:
    turn: int
    active: str           # the active question when the page was opened
    via: str              # "search" | "revisit"
    query: str | None = None


@dataclass
class PageNode:
    id: str               # alias shown to the model
    label: str            # real page label (never shown)
    image: Image.Image    # original resolution (shown resized per frame; the crop source)
    query: str            # the search that first found it
    descriptor: str | None = None
    facts: list[Fact] = field(default_factory=list)
    visits: list[Visit] = field(default_factory=list)
    next_fact: int = 1

    def fact(self, fact_id: str) -> Fact | None:
        return next((f for f in self.facts if f.id == fact_id), None)

    def copy(self) -> "PageNode":
        facts = [replace(f, bbox=list(f.bbox), history=list(f.history)) for f in self.facts]
        return replace(self, facts=facts, visits=list(self.visits))

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "query": self.query, "descriptor": self.descriptor,
                "size": list(self.image.size), "facts": [f.as_dict() for f in self.facts],
                "visits": [vars(v) for v in self.visits]}


@dataclass
class QNode:
    id: str
    question: str
    parent_id: str | None
    status: str = OPEN
    answer: str | None = None
    pages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "question": self.question, "parent_id": self.parent_id, "status": self.status,
                "answer": self.answer, "pages": list(self.pages)}


@dataclass
class Zoom:
    image: Image.Image                          # the crop at original resolution (shown resized per frame)
    region: tuple[int, int, int, int]           # crop rectangle in original page pixels (padded, clamped)
    turn: int                                   # turn that made it (names its image files)


@dataclass
class View:
    """What is in the model's visual context this turn: one page, optionally its zoom."""
    page: str
    via: str              # "search" | "revisit"
    query: str | None = None
    zoom: Zoom | None = None

    @property
    def kind(self) -> str:
        return "zoom" if self.zoom is not None else "page"


class Memory:
    def __init__(self, question: str) -> None:
        self.question = question
        self.nodes: list[QNode] = [QNode(ROOT_ID, question, None)]
        self.pages: dict[str, PageNode] = {}
        self.aliases: dict[str, str] = {}      # real label -> alias
        self.next_node = 1

    # ------------------------------------------------------------------ access

    @property
    def root(self) -> QNode:
        return self.nodes[0]

    @property
    def children(self) -> list[QNode]:
        return self.nodes[1:]

    def node(self, node_id: str) -> QNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    @property
    def active_id(self) -> str:
        """The first unresolved sub-question; the root when there is none (§2)."""
        return next((n.id for n in self.children if n.status != RESOLVED), ROOT_ID)

    def linked(self, page_id: str) -> list[str]:
        return [n.id for n in self.nodes if page_id in n.pages]

    def support_pages(self, node_ids: list[str]) -> list[str]:
        """Deduplicated union of the pages of the given nodes, in node order."""
        out: list[str] = []
        for nid in node_ids:
            node = self.node(nid)
            for p in node.pages if node is not None else []:
                if p not in out:
                    out.append(p)
        return out

    def evidence_pages(self) -> list[str]:
        """E_final (DESIGN.md §11.8): every page attached to a question, deduplicated in node order."""
        return self.support_pages([n.id for n in self.nodes])

    def add_page(self, label: str, image: Image.Image, query: str) -> PageNode:
        if label in self.aliases:
            return self.pages[self.aliases[label]]
        alias = f"P{len(self.pages) + 1}"
        page = PageNode(id=alias, label=label, image=image, query=query)
        self.pages[alias] = page
        self.aliases[label] = alias
        return page

    def clone(self) -> "Memory":
        """Deep copy of the editable state; page images are shared, never mutated."""
        m = Memory.__new__(Memory)
        m.question = self.question
        m.nodes = [replace(n, pages=list(n.pages)) for n in self.nodes]
        m.pages = {k: p.copy() for k, p in self.pages.items()}
        m.aliases = dict(self.aliases)
        m.next_node = self.next_node
        return m

    # ------------------------------------------------------------------ transitions

    def _new_node(self, question: str, **kw: Any) -> QNode:
        node = QNode(f"N{self.next_node}", question, ROOT_ID, **kw)
        self.next_node += 1
        return node

    def expand(self, op: Expand) -> list[str]:
        """The listed pages answer part of the active question. Root active: keep the root, append
        [answered (resolved, those pages), next (open)]. Sub-question active: it becomes the answered
        part and the remaining question is inserted after it."""
        pages = self.tree_pages(op.pages)
        active = self.node(self.active_id)
        if active.parent_id is None:
            done = self._new_node(op.answered, status=RESOLVED, answer=op.answer, pages=pages)
            nxt = self._new_node(op.next)
            self.nodes.extend([done, nxt])
            return [done.id, nxt.id]
        active.question, active.status, active.answer = op.answered, RESOLVED, op.answer
        active.pages.extend(p for p in pages if p not in active.pages)
        nxt = self._new_node(op.next)
        self.nodes.insert(self.nodes.index(active) + 1, nxt)
        return [active.id, nxt.id]

    def resolve(self, op: Resolve) -> str:
        """The listed pages answer the active question (or op.node_id): resolved, answer set, pages attached."""
        node = self.node(op.node_id or self.active_id)
        if node is None:
            raise EditError(f"unknown question {op.node_id}; existing: {', '.join(n.id for n in self.nodes)}")
        pages = self.tree_pages(op.pages)
        node.status, node.answer = RESOLVED, op.answer
        node.pages.extend(p for p in pages if p not in node.pages)
        return node.id

    def tree_pages(self, pages: list[str]) -> list[str]:
        """The pages a conclusion lists (DESIGN.md §11.23): each seen and with facts (this turn's included)."""
        for pid in pages:
            page = self.pages.get(pid)
            if page is None:
                raise EditError(f'"pages" names unknown page {pid}; pages seen: {", ".join(self.pages) or "none"}')
            if not page.facts:
                raise EditError(f'"pages" names {pid}, which has no facts; record what it shows first, '
                                f'and list only pages whose facts the answer rests on')
        return list(pages)

    def finish(self, answer: str) -> None:
        """The final answer resolves the original question (no page is attached by answering)."""
        self.root.status, self.root.answer = RESOLVED, answer

    def as_serializable(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "active": self.active_id,
            "nodes": [n.as_dict() for n in self.nodes],
            "pages": [p.as_dict() for p in self.pages.values()],
        }


# ---------------------------------------------------------------------- edits


def _to_page_frame(box: Box, view: View, page: PageNode, frame: Frame) -> tuple[Box, str]:
    """A box the model wrote on the image in view (in its frame) -> original page pixels."""
    if view.zoom is not None:
        unit = frame.to_unit(box, frame.shown_size(view.zoom.image.size))
        return unit_to_region(unit, view.zoom.region), "zoom"
    w, h = page.image.size
    return unit_to_region(frame.to_unit(box, frame.shown_size((w, h))), (0, 0, w, h)), "page"


def resolve_fact_id(fact_id: str, page_id: str) -> str:
    """Ops name facts of the page in view by their page-local id (F2); P1.F2 is accepted too."""
    return fact_id if "." in fact_id else f"{page_id}.{fact_id}"


def apply_edit(memory: Memory, edit: PageEdit | None, tree: Resolve | Expand | None, view: View | None, *,
               turn: int, frame: Frame) -> tuple[Memory, dict[str, Any]]:
    """Validate and apply the turn's edits on a clone: the page edit (desc, facts; boxes read in the
    model's `frame`, stored in original page pixels), then the tree edit, which attaches the pages it
    lists (DESIGN.md §11.23). Raises EditError (soft lane) without touching `memory`."""
    m = memory.clone()
    events: dict[str, Any] = {"page_desc": None, "added_facts": [], "updated_facts": [], "resolved": [], "expanded": []}
    if edit is not None and not edit.is_empty:
        _apply_page_edit(m, edit, view, turn, frame, events)
    if isinstance(tree, Resolve):
        events["resolved"].append(m.resolve(tree))
    elif isinstance(tree, Expand):
        events["expanded"].extend(m.expand(tree))
    return m, events


def _apply_page_edit(m: Memory, edit: PageEdit, view: View | None, turn: int, frame: Frame,
                     events: dict[str, Any]) -> None:
    if view is None:
        raise EditError("no page in view: there is nothing to describe or read")
    page = m.pages[view.page]
    if edit.desc is not None:
        page.descriptor = edit.desc
        events["page_desc"] = page.id
    for op in edit.facts:
        box, src = _to_page_frame(op.bbox, view, page, frame)
        if isinstance(op, AddFact):
            fact = Fact(f"{page.id}.F{page.next_fact}", op.text, box, src)
            page.next_fact += 1
            page.facts.append(fact)
            events["added_facts"].append(fact.id)
            continue
        fact_id = resolve_fact_id(op.fact_id, page.id)
        fact = page.fact(fact_id)
        if fact is None:
            owner = fact_id.split(".")[0]
            if owner in m.pages and owner != page.id:
                raise EditError(f"{op.fact_id} belongs to {owner}, not the page in view ({page.id}); revisit {owner} to correct it")
            raise EditError(f"unknown fact {op.fact_id}; facts of {page.id}: "
                            f"{', '.join(short_fact_id(f.id) for f in page.facts) or 'none'}")
        _correct(fact, op.text, box, src, turn)
        events["updated_facts"].append(fact.id)


def _correct(fact: Fact, text: str, box: Box, src: str, turn: int) -> None:
    fact.history.append({"turn": turn, "text": fact.text, "bbox": list(fact.bbox), "src": fact.src})
    fact.text, fact.bbox, fact.src = text, box, src
    fact.revision += 1


# ---------------------------------------------------------------------- rendering


def short_fact_id(fact_id: str) -> str:
    """P1.F2 -> F2 (facts are shown inside their page's block)."""
    return fact_id.split(".", 1)[1] if "." in fact_id else fact_id


def _node_head(node: QNode, active: str) -> str:
    tag = node.status + (", ACTIVE" if node.id == active else "")
    return f"{node.id} [{tag}]: {node.question}"


def _node_body(node: QNode) -> list[str]:
    rows = []
    if node.answer is not None:
        rows.append(f"answer: {node.answer}")
    if node.pages:
        rows.append(f"pages: [{', '.join(node.pages)}]")
    return rows


def render_tree(m: Memory) -> str:
    """Question tree: N0 on top, sub-questions below in order; no boxes, no page labels."""
    active = m.active_id
    rows = ["Question tree:", _node_head(m.root, active), *_node_body(m.root)]
    kids = m.children
    for i, node in enumerate(kids):
        last = i == len(kids) - 1
        rows.append("│")
        rows.append(("└─ " if last else "├─ ") + _node_head(node, active))
        rows.extend(("   " if last else "│  ") + line for line in _node_body(node))
    return "\n".join(rows)


def render_pages(m: Memory) -> str:
    """Page memory: one block per page seen, its facts by page-local id (F1, F2, ...)."""
    if not m.pages:
        return "Page memory: none"
    rows = ["Page memory:"]
    for i, p in enumerate(m.pages.values()):
        if i:
            rows.append("")
        head = p.id + (f": {p.descriptor}" if p.descriptor else "")
        linked = m.linked(p.id)
        tail = f"   -> answers {', '.join(linked)}" if linked else ""
        rows.append(f"{head}   (found by: {json.dumps(p.query, ensure_ascii=False)}){tail}")
        for j, f in enumerate(p.facts):
            rows.append(("└─ " if j == len(p.facts) - 1 else "├─ ") + f"{short_fact_id(f.id)}: {f.text}")
    return "\n".join(rows)


def box_rule(frame: Frame, image: str, size: tuple[int, int]) -> str:
    """How boxes on `image` ("PAGE" / "ZOOM", shown at `size`) are written in this frame."""
    if frame.coord == "px":
        return f"ABSOLUTE PIXELS on the {image} image ({size[0]} wide x {size[1]} high)"
    return f"0-1000 RELATIVE coordinates on the {image} image"


ANSWER_ONLY_HINT = ('Your last response could not be used. Answer now from your page memory, with no page edit:\n'
                    '<page>null</page>\n<tree>null</tree>\n<act>{"type": "answer", "text": "<final answer>"}</act>')


def turn_hint(m: Memory, view: View | None, frame: Frame, *, force: bool = False, answer_only: bool = False,
              turn_of: tuple[int, int] | None = None) -> str:
    """The per-turn hint. turn_of = (k, n) opens it with "Turn k of n" (DESIGN.md §11.24), else "This turn"."""
    if answer_only:
        return ANSWER_ONLY_HINT
    head = f"Turn {turn_of[0]} of {turn_of[1]}" if turn_of else "This turn"
    if view is None:
        hint = f"{head}: no page in view -> <page>null</page>; action: search, revisit or answer."
    elif view.zoom is None:
        rule = box_rule(frame, "PAGE", frame.shown_size(m.pages[view.page].image.size))
        hint = f"{head}: page {view.page} is in view. Boxes in facts and crop are {rule}. You may crop this page once."
    else:
        rule = box_rule(frame, "ZOOM", frame.shown_size(view.zoom.image.size))
        hint = f"{head}: page {view.page} and its zoom are in view. Boxes in facts are {rule}. No further crop on this visit."
    if force:
        hint += ("\nBudget exhausted: this is your last turn. You may still edit the memory; the action must be answer,"
                 " stating only what your page memory supports.")
    return hint


def render_state(m: Memory, view: View | None, *, frame: Frame, note: str = "", force: bool = False,
                 answer_only: bool = False, turn_of: tuple[int, int] | None = None) -> str:
    lines = [render_tree(m), "", render_pages(m), ""]
    lines.append("Page in view: none" if view is None else f"Page in view: {view.page}" + (" + zoom" if view.zoom else ""))
    if note:
        lines.append(f"Note: {note}")
    lines.append(turn_hint(m, view, frame, force=force, answer_only=answer_only, turn_of=turn_of))
    return "\n".join(lines)
