from __future__ import annotations

"""AGv2 merged-action protocol: the ONE parser + state helpers shared by eval (src/agent.py)
and RL (verl-agent slidevqa envs.py).

Response format (every policy turn):

    <think>...</think>
    <update_graph>{...}</update_graph>     # required iff an observation is pending
    <search>query</search> | <bbox>[x1,y1,x2,y2]</bbox> | <answer>final</answer>

Structural rules (violations are format errors -> episode terminates on both sides):
  - response starts with a non-empty <think> block;
  - exactly one action tag among <search>/<bbox>/<answer>;
  - <update_graph> appears iff an observation is pending, between </think> and the action tag;
  - no stray text outside the blocks (whitespace ok);
  - <bbox> requires a crop target (set by the accept/expand commit of a page).

Crop semantics (see DESIGN_AGV2.md):
  - a page commit with accept/expand records (crop_source_image, crop_target_node);
  - <bbox> always crops the SOURCE PAGE image (never a crop of a crop);
  - the next turn's <update_graph> with accept/expand APPENDS its facts to crop_target_node
    (no structural graph change); reject discards.
"""

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


THINK_RE = re.compile(r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL)
UPDATE_RE = re.compile(r"<\s*update_graph\s*>(.*?)<\s*/\s*update_graph\s*>", re.IGNORECASE | re.DOTALL)
ACTION_TAGS = ("search", "bbox", "answer")


@dataclass
class ParsedTurn:
    think: str
    update_payload: dict[str, Any] | None  # parsed JSON of <update_graph>, None if absent
    action: str | None                     # "search" | "bbox" | "answer" | None (commit-only)
    action_payload: str                    # raw text inside the action tag ("" if commit-only)
    box: list[float] | None = None         # parsed for action == "bbox"


class ProtocolError(ValueError):
    """Structural violation of the AGv2 response format (format_error lane)."""


class BoxFormatError(ValueError):
    """<bbox> payload malformed (not 4 numbers / not a valid rectangle). Soft error lane."""


def parse_turn(text: str, *, observation_pending: bool) -> ParsedTurn:
    """Parse one model response under AGv2 rules. Raises ProtocolError on structural
    violations and BoxFormatError when only the bbox payload itself is malformed."""
    stripped = (text or "").strip()

    think_match = THINK_RE.match(stripped)
    if think_match is None or think_match.start() != 0:
        raise ProtocolError("response must start with <think>...</think>")
    think = think_match.group(1).strip()
    if not think:
        raise ProtocolError("<think> must be non-empty")
    rest = stripped[think_match.end():].strip()

    update_matches = list(UPDATE_RE.finditer(rest))
    if len(update_matches) > 1:
        raise ProtocolError("at most one <update_graph> block is allowed")
    update_payload: dict[str, Any] | None = None
    if update_matches:
        m = update_matches[0]
        if rest[: m.start()].strip():
            raise ProtocolError("<update_graph> must come directly after </think>")
        if not observation_pending:
            raise ProtocolError("<update_graph> is only allowed when an observation is pending")
        raw = m.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"<update_graph> is not valid JSON: {exc.msg}")
        if not isinstance(parsed, dict):
            raise ProtocolError("<update_graph> must be a JSON object")
        update_payload = parsed
        rest = rest[m.end():].strip()
    elif observation_pending:
        raise ProtocolError(
            "an observation is pending: the response must commit it with <update_graph> "
            "before the action tag"
        )

    matches: list[tuple[int, int, str, str]] = []
    for tag in ACTION_TAGS:
        for m in re.finditer(
            rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>", rest, re.IGNORECASE | re.DOTALL
        ):
            matches.append((m.start(), m.end(), tag, m.group(1).strip()))
    if not matches:
        # Commit-only turn: legal iff an <update_graph> was given — the model defers its
        # action one turn to see the UPDATED graph rendered (typically before the final
        # answer of a multi-hop question).
        if update_payload is None:
            raise ProtocolError("expected exactly one action tag, got none")
        if rest.strip():
            raise ProtocolError("no text is allowed outside the tag blocks")
        return ParsedTurn(
            think=think, update_payload=update_payload, action=None, action_payload=""
        )
    if len(matches) != 1:
        tags = sorted(m[2] for m in matches)
        raise ProtocolError(f"expected exactly one action tag, got {tags}")
    start, end, action, payload = matches[0]
    if rest[:start].strip() or rest[end:].strip():
        raise ProtocolError("no text is allowed outside the tag blocks")

    box: list[float] | None = None
    if action == "bbox":
        box = parse_box(payload)
    elif action == "search":
        if not payload:
            raise ProtocolError("<search> requires a non-empty query")
    elif action == "answer":
        if not payload:
            raise ProtocolError("<answer> requires a non-empty final answer")

    return ParsedTurn(
        think=think,
        update_payload=update_payload,
        action=action,
        action_payload=payload,
        box=box,
    )


def parse_box(payload: str) -> list[float]:
    """Parse a <bbox> payload into [x1,y1,x2,y2]. Raises BoxFormatError (soft lane)."""
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?", payload or "")]
    if not isinstance(parsed, list) or len(parsed) != 4:
        raise BoxFormatError("bbox must contain exactly four numbers [x1,y1,x2,y2]")
    try:
        box = [float(v) for v in parsed]
    except (TypeError, ValueError):
        raise BoxFormatError("bbox values must be numbers")
    if any(v < 0 for v in box):
        raise BoxFormatError("bbox values must be non-negative")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise BoxFormatError("bbox must satisfy x2 > x1 and y2 > y1")
    return box


def crop_displayed_box(
    raw_image: Image.Image,
    displayed_size: tuple[int, int],
    box: list[float],
    *,
    pad: int = 28,
    min_pixels: int | None = None,
) -> Image.Image:
    """UniDoc-style crop: `box` is absolute pixels in the DISPLAYED image space; map linearly
    to the raw image, expand by `pad` raw pixels, clamp, crop, then smart-resize the crop up
    to at least min_pixels (32-aligned) so fine print is actually readable."""
    raw_w, raw_h = raw_image.size
    disp_w, disp_h = displayed_size
    if disp_w <= 0 or disp_h <= 0:
        raise BoxFormatError(f"invalid displayed size: {displayed_size}")
    x1, y1, x2, y2 = box
    rx1 = x1 * raw_w / disp_w
    ry1 = y1 * raw_h / disp_h
    rx2 = x2 * raw_w / disp_w
    ry2 = y2 * raw_h / disp_h
    rx1 = max(rx1 - pad, 0)
    ry1 = max(ry1 - pad, 0)
    rx2 = min(rx2 + pad, raw_w)
    ry2 = min(ry2 + pad, raw_h)
    left, upper = math.floor(rx1), math.floor(ry1)
    right, lower = math.ceil(rx2), math.ceil(ry2)
    if right <= left or lower <= upper:
        raise BoxFormatError(f"bbox maps to an empty region: {box}")
    cropped = raw_image.convert("RGB").crop((left, upper, right, lower))
    if min(cropped.size) <= 0 or max(cropped.size) / max(1, min(cropped.size)) >= 180.0:
        raise BoxFormatError(f"crop aspect ratio too extreme for bbox: {box}")
    return smart_resize(cropped, min_pixels=min_pixels)


@dataclass
class CropGeometry:
    """How a crop image maps back onto its source page (AGv2.1 grounded facts).
    region_raw is the crop rectangle in RAW page pixels (post-pad, post-clamp);
    crop_size is the final (smart-resized) crop image size the model actually views."""
    region_raw: tuple[float, float, float, float]
    crop_size: tuple[int, int]
    page_raw_size: tuple[int, int]
    page_displayed_size: tuple[int, int]


def crop_displayed_box_with_geom(
    raw_image: Image.Image,
    displayed_size: tuple[int, int],
    box: list[float],
    *,
    pad: int = 28,
    min_pixels: int | None = None,
) -> tuple[Image.Image, "CropGeometry"]:
    """crop_displayed_box + the geometry needed to remap crop-frame boxes to the page."""
    raw_w, raw_h = raw_image.size
    disp_w, disp_h = displayed_size
    if disp_w <= 0 or disp_h <= 0:
        raise BoxFormatError(f"invalid displayed size: {displayed_size}")
    x1, y1, x2, y2 = box
    rx1 = max(x1 * raw_w / disp_w - pad, 0)
    ry1 = max(y1 * raw_h / disp_h - pad, 0)
    rx2 = min(x2 * raw_w / disp_w + pad, raw_w)
    ry2 = min(y2 * raw_h / disp_h + pad, raw_h)
    left, upper = math.floor(rx1), math.floor(ry1)
    right, lower = math.ceil(rx2), math.ceil(ry2)
    if right <= left or lower <= upper:
        raise BoxFormatError(f"bbox maps to an empty region: {box}")
    cropped = raw_image.convert("RGB").crop((left, upper, right, lower))
    if min(cropped.size) <= 0 or max(cropped.size) / max(1, min(cropped.size)) >= 180.0:
        raise BoxFormatError(f"crop aspect ratio too extreme for bbox: {box}")
    final = smart_resize(cropped, min_pixels=min_pixels)
    geom = CropGeometry(
        region_raw=(float(left), float(upper), float(right), float(lower)),
        crop_size=final.size,
        page_raw_size=(raw_w, raw_h),
        page_displayed_size=(int(disp_w), int(disp_h)),
    )
    return final, geom


def map_crop_box_to_page(box: list[float], geom: "CropGeometry") -> list[float]:
    """Map a box given in crop-image pixels back to source-page DISPLAYED pixels."""
    left, upper, right, lower = geom.region_raw
    cw, ch = geom.crop_size
    raw_w, raw_h = geom.page_raw_size
    disp_w, disp_h = geom.page_displayed_size
    sx = (right - left) / max(cw, 1)
    sy = (lower - upper) / max(ch, 1)
    rx1, ry1 = left + box[0] * sx, upper + box[1] * sy
    rx2, ry2 = left + box[2] * sx, upper + box[3] * sy
    out = [
        rx1 * disp_w / raw_w,
        ry1 * disp_h / raw_h,
        rx2 * disp_w / raw_w,
        ry2 * disp_h / raw_h,
    ]
    return [round(v, 1) for v in out]


def remap_crop_decision_boxes(decision: Any, geom: "CropGeometry | None") -> None:
    """Remap every grounded fact on a CROP commit from crop-frame to page-frame pixels,
    in place. Without geometry the boxes are unmappable -> stripped (fact kept)."""
    for f in getattr(decision, "supporting_facts", None) or []:
        box = getattr(f, "bbox_2d", None)
        if not box:
            continue
        f.bbox_2d = map_crop_box_to_page(box, geom) if geom is not None else None


def smart_resize(image: Image.Image, *, factor: int = 32, min_pixels: int | None = None) -> Image.Image:
    import os
    if min_pixels is None:
        min_pixels = int(os.environ.get("ACTIVE_GRAPH_MIN_PIXELS", "500000"))
    width, height = image.size
    if width <= 0 or height <= 0:
        return image.convert("RGB").copy()
    new_width = max(factor, round(width / factor) * factor)
    new_height = max(factor, round(height / factor) * factor)
    if new_width * new_height < min_pixels:
        scale = math.sqrt(min_pixels / float(new_width * new_height))
        new_width = max(factor, math.ceil(new_width * scale / factor) * factor)
        new_height = max(factor, math.ceil(new_height * scale / factor) * factor)
    if (new_width, new_height) == (width, height):
        return image.convert("RGB")
    return image.convert("RGB").resize((new_width, new_height), Image.Resampling.BICUBIC)


@dataclass
class CropContext:
    """Where <bbox> crops from and where crop-commit facts go. Armed by page commits with
    accept/expand; cleared by reject. While a crop chain is live, the graph's active node is
    PINNED to target_node_id (the crop is still work on that node's question); the move the
    accept/expand would normally make (accept -> parent, expand -> remaining child) is stored
    in resume_active_node_id and executed by finish_crop_chain() when the chain ends."""
    source_image: Any | None = None      # raw PIL image of the committed page
    displayed_size: tuple[int, int] | None = None
    source_page_label: str = ""
    target_node_id: str | None = None
    resume_active_node_id: str | None = None
    # geometry of the most recent crop taken from this page (AGv2.1): used to remap fact
    # bbox_2d emitted while viewing the crop back into source-page displayed pixels.
    geometry: Any | None = None

    @property
    def ready(self) -> bool:
        return self.source_image is not None and self.target_node_id is not None

    def clear(self) -> None:
        self.source_image = None
        self.displayed_size = None
        self.source_page_label = ""
        self.target_node_id = None
        self.resume_active_node_id = None
        self.geometry = None


def commit_page_decision(
    graph: Any,
    decision: Any,
    observation_summary: str = "",
    *,
    defer_active_shift: bool = False,
) -> tuple[str, str | None, str | None]:
    """Apply a page-observation <update_graph> to the graph at the active node.
    Returns (decision_type_lower, facts_target_node_id or None, resume_active_node_id or None).

    facts_target_node_id = the node that received this page's facts (accept -> the node
    itself; expand -> the answered child; reject -> None) = the crop target.

    defer_active_shift: set True when the SAME turn's action is <bbox> — the upcoming crop is
    still work on the facts node, so active is pinned back to it and the destination the
    commit would have moved to (accept -> parent, expand -> remaining child) is returned as
    resume_active_node_id for finish_crop_chain()."""
    from src.active_clue_graph import apply_graph_decision

    node_id = graph.active().id
    dtype = str(getattr(decision, "type", "")).upper()
    apply_graph_decision(
        graph=graph, node_id=node_id, decision=decision, observation_summary=observation_summary
    )
    if dtype == "ACCEPT":
        facts_node = node_id
    elif dtype == "EXPAND":
        # apply_graph_decision created [answered(sufficient), remaining(open)] and set active
        # to remaining; the answered child (facts holder) is its sibling.
        node = graph.nodes[node_id]
        answered = [cid for cid in node.children if cid != graph.active_node_id]
        facts_node = answered[-1] if answered else node_id
    else:
        return "reject", None, None

    resume_id: str | None = None
    if defer_active_shift:
        resume_id = graph.active_node_id      # where the commit moved active to
        graph.set_active(facts_node)          # pin active on the node being zoomed
    return dtype.lower(), facts_node, resume_id


def finish_crop_chain(graph: Any, crop_ctx: "CropContext") -> None:
    """End a crop chain: execute the deferred active move and clear the context."""
    if crop_ctx.resume_active_node_id and crop_ctx.resume_active_node_id in graph.nodes:
        graph.set_active(crop_ctx.resume_active_node_id)
    crop_ctx.clear()


def commit_crop_decision(graph: Any, target_node_id: str, decision: Any) -> str:
    """Apply a crop-observation <update_graph>: accept/expand append the facts to the crop
    target node (no structural change, no status change, no active change); reject discards.
    Returns the decision type (lower)."""
    from src.active_clue_graph import _compact_new, _fact_entry

    dtype = str(getattr(decision, "type", "")).upper()
    if dtype in {"ACCEPT", "EXPAND"}:
        node = graph.nodes.get(target_node_id)
        if node is None:
            raise ProtocolError(f"crop target node {target_node_id!r} no longer exists")
        facts = [_fact_entry(f) for f in (getattr(decision, "supporting_facts", None) or [])]
        if not facts and getattr(decision, "answer", None):
            facts = [_fact_entry(str(decision.answer))]
        node.known_facts.extend(_compact_new(node.known_facts, facts))
        refs = list(getattr(decision, "evidence_refs", None) or [])
        node.evidence_refs.extend(_compact_new(node.evidence_refs, refs))
        # the zoom refines the node's answer (e.g. "approximately 40%" -> "42%")
        answer = getattr(decision, "answer", None)
        if answer is not None and str(answer).strip():
            node.answer = str(answer).strip()
        return dtype.lower()
    if dtype == "REJECT":
        return "reject"
    raise ProtocolError(f"unknown crop commit type: {dtype!r}")


def finalize_root(graph: Any, answer: str) -> None:
    """<answer> implicitly finalizes the root: mark it sufficient and store the final answer.
    Root facts are left as-is (children hold the evidence); if the root has neither facts nor
    sufficient children, this is still allowed (single-search direct answers)."""
    from src.active_clue_graph import SUFFICIENT_STATUS

    root = graph.root()
    root.status = SUFFICIENT_STATUS
    root.answer = str(answer).strip()


def open_nonroot_nodes(graph: Any) -> list[str]:
    """Node ids (excluding root) still open — used by the multi-hop answer gate."""
    return [
        node_id
        for node_id, node in graph.nodes.items()
        if node_id != graph.root_id and node.status == "open"
    ]


def pending_hint(
    observation_pending: bool,
    *,
    crop_page_label: str = "",
    crop_target_node: str = "",
) -> str:
    if observation_pending and crop_target_node:
        label = crop_page_label or "the committed page"
        return (
            f"Pending ZOOM observation of {label}: commit it with <update_graph> "
            f"(accept/expand adds its facts to node [{crop_target_node}]; reject discards), "
            "then give exactly one action."
        )
    if observation_pending:
        return (
            "Pending observation: you MUST commit it first with <update_graph>, "
            "then give exactly one action."
        )
    return "No pending observation: do NOT emit <update_graph> this turn."
