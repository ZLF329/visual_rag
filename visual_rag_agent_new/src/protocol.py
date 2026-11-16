from __future__ import annotations

"""new protocol: the ONE schema + parser + coordinate frames + crop geometry shared by eval
(src/agent.py) and RL (rl/envs.py). See DESIGN.md §3-§4 and §11 (frames and crop: §11.3).

Every turn (DESIGN.md §11.23, four blocks since §11.24):
    <think>...</think>
    <page>{"desc": ..., "facts": [...]} | null</page>
    <tree>{"op": "resolve" | "expand", ...} | null</tree>
    <act>{"type": ...}</act>
<page> is what the page in view gives (its description and readings); <tree> is a conclusion for a
question (resolve / expand), drawn from the page memory, that attaches exactly the pages it lists;
<act> is the one action. Each block holds one JSON value, so a slip in one cannot swallow the others.

ProtocolError = hard lane: unparseable, schema violation or structural rule; the episode ends.
EditError     = soft lane: well-formed, but it cannot be applied to the current memory or view
                (unknown ids, unusable boxes). Nothing is applied and the model retries.
"""

import json
import math
import re
from dataclasses import dataclass, replace
from typing import Annotated, Literal, Union

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
BLOCKS = ("page", "tree", "act")       # the JSON blocks after the think, in canonical order
BLOCK_RE = {tag: re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL) for tag in BLOCKS}

ViewKind = Literal["none", "page", "zoom"]
Box = list[float]
MIN_BOX_SIDE = 4.0  # shown-image pixels; anything smaller is not a readable region
# SCORE / VISOR image budget: every image a model sees (page or crop) has this many pixels.
IMAGE_MIN_PIXELS = 256 * 28 * 28
IMAGE_MAX_PIXELS = 512 * 28 * 28
CROP_PAD = 28       # original-page pixels added on each side of a crop box (SCORE App. B)
MAX_ASPECT = 180.0  # Qwen's processors reject aspect ratios above 200


class ProtocolError(ValueError):
    """Structural violation of the turn format (hard lane)."""


class EditError(ValueError):
    """A well-formed turn that cannot be applied to the current memory / view (soft lane)."""


class BoxError(EditError):
    """A box is well-formed but geometrically unusable on the image in view (soft lane)."""


# ------------------------------------------------------------------ schema


def _box4(value: object) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must be a list of exactly four numbers")
    out: Box = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("bbox values must be numbers")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError("bbox values must be finite")
        out.append(f)
    return out


def _text(value: object, name: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _ref(value: object, name: str) -> str:
    """Page / fact / node ids are case-insensitive (P1, F2 or P1.F2, N3)."""
    return _text(value, name).upper()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _pages(value: object) -> list[str]:
    """The pages a conclusion rests on (DESIGN.md §11.23): at least one page id, deduplicated in order."""
    if not isinstance(value, list) or not value:
        raise ValueError('"pages" must list at least one page id, e.g. ["P1"]')
    out: list[str] = []
    for item in value:
        ref = _ref(item, "pages")
        if ref not in out:
            out.append(ref)
    return out


class AddFact(_Strict):
    op: Literal["add_fact"]
    text: str
    bbox: Box

    @field_validator("text")
    @classmethod
    def _t(cls, v: object) -> str:
        return _text(v, "text")

    @field_validator("bbox", mode="before")
    @classmethod
    def _b(cls, v: object) -> Box:
        return _box4(v)


class UpdateFact(_Strict):
    op: Literal["update_fact"]
    fact_id: str
    text: str
    bbox: Box

    @field_validator("fact_id")
    @classmethod
    def _id(cls, v: object) -> str:
        return _ref(v, "fact_id")

    @field_validator("text")
    @classmethod
    def _t(cls, v: object) -> str:
        return _text(v, "text")

    @field_validator("bbox", mode="before")
    @classmethod
    def _b(cls, v: object) -> Box:
        return _box4(v)


FactOp = Annotated[Union[AddFact, UpdateFact], Field(discriminator="op")]


class Resolve(_Strict):
    """The listed pages answer the active question (or the question `node_id`)."""
    op: Literal["resolve"]
    answer: str
    node_id: str | None = None
    pages: list[str]

    @field_validator("answer")
    @classmethod
    def _a(cls, v: object) -> str:
        return _text(v, "answer")

    @field_validator("node_id", mode="before")
    @classmethod
    def _n(cls, v: object) -> str | None:
        return None if v is None else _ref(v, "node_id")

    @field_validator("pages", mode="before")
    @classmethod
    def _p(cls, v: object) -> list[str]:
        return _pages(v)


class Expand(_Strict):
    """The listed pages answer part of the active question: the answered part + the remaining question."""
    op: Literal["expand"]
    answered: str
    answer: str
    next: str
    pages: list[str]

    @field_validator("answered", "answer", "next")
    @classmethod
    def _t(cls, v: object, info) -> str:
        return _text(v, info.field_name)

    @field_validator("pages", mode="before")
    @classmethod
    def _p(cls, v: object) -> list[str]:
        return _pages(v)


TreeEdit = Annotated[Union[Resolve, Expand], Field(discriminator="op")]


class PageEdit(_Strict):
    """What the page in view gives this turn: its description and its readings."""
    desc: str | None = None
    facts: list[FactOp] = Field(default_factory=list)

    @field_validator("desc", mode="before")
    @classmethod
    def _d(cls, v: object) -> str | None:
        return None if v is None else _text(v, "desc")

    @property
    def is_empty(self) -> bool:
        return self.desc is None and not self.facts


class SearchAction(_Strict):
    type: Literal["search"]
    query: str

    @field_validator("query")
    @classmethod
    def _q(cls, v: object) -> str:
        return _text(v, "query")


class RevisitAction(_Strict):
    type: Literal["revisit"]
    page_id: str

    @field_validator("page_id")
    @classmethod
    def _p(cls, v: object) -> str:
        return _ref(v, "page_id")


class CropAction(_Strict):
    """Zoom into the page in view: a box in PAGE pixels, or the region of one of its facts."""
    type: Literal["crop"]
    bbox: Box | None = None
    fact_id: str | None = None

    @field_validator("bbox", mode="before")
    @classmethod
    def _b(cls, v: object) -> Box | None:
        return None if v is None else _box4(v)

    @field_validator("fact_id", mode="before")
    @classmethod
    def _f(cls, v: object) -> str | None:
        return None if v is None else _ref(v, "fact_id")

    @model_validator(mode="after")
    def _one_target(self) -> "CropAction":
        if (self.bbox is None) == (self.fact_id is None):
            raise ValueError('crop takes exactly one of "bbox" or "fact_id"')
        return self


class AnswerAction(_Strict):
    type: Literal["answer"]
    text: str

    @field_validator("text")
    @classmethod
    def _t(cls, v: object) -> str:
        return _text(v, "text")


Action = Annotated[Union[SearchAction, RevisitAction, CropAction, AnswerAction], Field(discriminator="type")]


class Act(_Strict):
    page: PageEdit | None = None
    tree: TreeEdit | None = None
    action: Action

    def to_payload(self) -> dict:
        """Canonical shape (also the trace record): "page", "tree", "action" always present, in that order; a
        page edit with both keys; crop carries only its target; resolve without node_id omits it. Integral box
        values are written as integers."""
        page = None
        if self.page is not None:
            facts = [op.model_dump(mode="json") for op in self.page.facts]
            for op in facts:
                op["bbox"] = _box_json(op["bbox"])
            page = {"desc": self.page.desc, "facts": facts}
        tree = None if self.tree is None else self.tree.model_dump(mode="json", exclude_none=True)
        action = self.action.model_dump(mode="json", exclude_none=self.action.type == "crop")
        if action.get("bbox") is not None:
            action["bbox"] = _box_json(action["bbox"])
        return {"page": page, "tree": tree, "action": action}

    def to_text(self) -> str:
        """The canonical <page>, <tree>, <act> blocks: compact JSON, null for an absent page or tree edit."""
        payload = self.to_payload()
        values = {"page": payload["page"], "tree": payload["tree"], "act": payload["action"]}
        return "\n".join(f"<{tag}>{json.dumps(values[tag], ensure_ascii=False, separators=(',', ':'))}</{tag}>"
                         for tag in BLOCKS)


def _box_json(box: list[float]) -> list[float | int]:
    return [int(v) if float(v).is_integer() else v for v in box]


@dataclass
class Turn:
    think: str
    act: Act


# ------------------------------------------------------------------ parsing


def _split_blocks(text: str, tags: tuple[str, ...] = BLOCKS) -> tuple[str, dict[str, str | None]]:
    """Lenient split (DESIGN.md §11.11): the think is every <think>...</think> block joined (may be empty);
    each of <page>, <tree>, <act> is the LAST such block outside the thinks (None when absent). Stray tags
    and text outside the blocks are ignored, but an opened block that is never closed is an error, not a
    silently dropped edit. `tags`: the blocks read (the others are ignored entirely)."""
    stripped = (text or "").strip()
    thinks = [t.strip() for t in THINK_RE.findall(stripped)]
    rest = THINK_RE.sub("", stripped)
    found: dict[str, str | None] = dict.fromkeys(BLOCKS)
    for tag in tags:
        hits = BLOCK_RE[tag].findall(rest)
        if not hits and f"<{tag}>" in rest:
            raise ProtocolError(f"<{tag}> is not closed with </{tag}>")
        found[tag] = hits[-1].strip() if hits else None
    if found["act"] is None:
        raise ProtocolError("missing <act>{...}</act> block")
    return "\n".join(t for t in thinks if t), found


def _decode_block(tag: str, inner: str | None) -> dict | None:
    """The first complete JSON value of a block; trailing characters (an extra brace) are ignored. <page> and
    <tree> may be null or left out; <act> must be an object."""
    if inner is None:
        return None
    try:
        value, _end = json.JSONDecoder().raw_decode(inner)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"<{tag}> is not valid JSON: {exc.msg}")
    if value is None and tag != "act":
        return None
    if not isinstance(value, dict):
        raise ProtocolError(f"<{tag}> must be a JSON object" + ("" if tag == "act" else " or null"))
    return value


_TAG_OF = {"page": "page", "tree": "tree", "action": "act"}


def _validation_summary(exc: ValidationError) -> str:
    """'<page> schema: facts.0.bbox: ...' - the block first, so a retry knows which block to fix."""
    parts = []
    for err in exc.errors()[:3]:
        loc = [str(p) for p in err.get("loc", ())]
        tag = _TAG_OF.get(loc[0], "act") if loc else "act"
        where = ".".join(loc[1:])
        parts.append(f"<{tag}> schema: " + (f"{where}: {err.get('msg')}" if where else str(err.get("msg"))))
    return "; ".join(parts)


def check_rules(act: Act, *, view: ViewKind, current_page: str | None = None, force_answer: bool = False,
                answer_only: bool = False) -> None:
    """Structural rules that depend on the view (DESIGN.md §3.1, §4; answer-only turn §11.11). The tree
    edit draws on the page memory, so it needs no page in view (§11.23). On the answer-only turn only the
    action is read (parse_turn drops the page and tree blocks, §11.26), so it just has to be an answer."""
    edit, action = act.page, act.action
    if (force_answer or answer_only) and action.type != "answer":
        raise ProtocolError("budget exhausted: the action must be answer")
    if view == "none":
        if edit is not None and not edit.is_empty:
            raise ProtocolError('no page in view: "page" must be null')
        if action.type == "crop":
            raise ProtocolError("no page in view: nothing to crop")
    if action.type == "crop" and view == "zoom":
        raise ProtocolError("one crop per visit: search, revisit another page or answer")
    if action.type == "revisit" and current_page is not None and action.page_id == current_page.upper():
        raise ProtocolError(f"{action.page_id} is already in view: revisit another page")


def parse_turn(text: str, *, view: ViewKind, current_page: str | None = None, force_answer: bool = False,
               answer_only: bool = False) -> Turn:
    # Answer-only turn (DESIGN.md §11.26): only <act> counts; a <page> or <tree> block written anyway - even
    # unclosed or malformed - is dropped, so a slip there never loses the answer.
    think, found = _split_blocks(text, tags=("act",) if answer_only else BLOCKS)
    payload = {"page": _decode_block("page", found["page"]), "tree": _decode_block("tree", found["tree"]),
               "action": _decode_block("act", found["act"])}
    try:
        act = Act.model_validate(payload)
    except ValidationError as exc:
        raise ProtocolError(_validation_summary(exc))
    check_rules(act, view=view, current_page=current_page, force_answer=force_answer, answer_only=answer_only)
    return Turn(think=think, act=act)


def render_turn(think: str, act: Act) -> str:
    """The canonical response text (SFT targets and the previous-response slot): one think block, then
    the <page>, <tree>, <act> blocks with canonical JSON."""
    return "<think>" + think + "</think>\n" + act.to_text()


# ------------------------------------------------------------------ frames & geometry
#
# Boxes are stored in ORIGINAL page pixels. A model reads and writes boxes in its own native frame
# (DESIGN.md §11.3): Qwen2.5-VL in absolute pixels of the image as shown to it, Qwen3-VL / 3.5 / 3.8
# in 0-1000 relative coordinates. Conversions go through unit boxes ([0, 1] fractions of an image).


def qwen_smart_resize(width: int, height: int, factor: int,
                      min_pixels: int = IMAGE_MIN_PIXELS, max_pixels: int = IMAGE_MAX_PIXELS) -> tuple[int, int]:
    """Qwen-VL's smart_resize: sides on the `factor` grid, pixel count within [min, max]."""
    if max(width, height) / max(1, min(width, height)) > 200:
        raise BoxError(f"image {width}x{height} has an extreme aspect ratio")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return w_bar, h_bar


@dataclass(frozen=True)
class Frame:
    """A model family's native box convention, vision patch grid and image budget."""
    name: str
    coord: Literal["px", "norm1000"]   # px: absolute pixels of the shown image; norm1000: 0-1000 relative
    factor: int                         # side multiple of the vision encoder (patch size x spatial merge)
    min_pixels: int = IMAGE_MIN_PIXELS
    max_pixels: int = IMAGE_MAX_PIXELS

    def shown_size(self, size: tuple[int, int]) -> tuple[int, int]:
        """The size at which an image of `size` is shown to this model (its exact input frame)."""
        return qwen_smart_resize(int(size[0]), int(size[1]), self.factor, self.min_pixels, self.max_pixels)

    def with_max_pixels(self, max_pixels: int) -> "Frame":
        """The same frame with a smaller image budget (DESIGN.md §11.25: the teacher sees small pages, so small
        print is genuinely hard to read and the zoom earns its turn). Every image, page or zoom, is fitted to it."""
        return replace(self, max_pixels=int(max_pixels), min_pixels=min(self.min_pixels, int(max_pixels)))

    def show(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        size = self.shown_size(image.size)
        return image if size == image.size else image.resize(size, Image.Resampling.BICUBIC)

    def _scale(self, shown: tuple[int, int]) -> tuple[float, float]:
        return (float(shown[0]), float(shown[1])) if self.coord == "px" else (1000.0, 1000.0)

    def to_unit(self, box: Box, shown: tuple[int, int]) -> Box:
        """A box in this frame (on an image shown at `shown`) -> clamped unit box. BoxError if less
        than MIN_BOX_SIDE shown pixels are left each way."""
        sx, sy = self._scale(shown)
        x1, y1, x2, y2 = (float(v) for v in box)
        cx1, cx2 = min(max(x1, 0.0), sx), min(max(x2, 0.0), sx)
        cy1, cy2 = min(max(y1, 0.0), sy), min(max(y2, 0.0), sy)
        if (cx2 - cx1) * shown[0] / sx < MIN_BOX_SIDE or (cy2 - cy1) * shown[1] / sy < MIN_BOX_SIDE:
            where = f"a {shown[0]}x{shown[1]} image" if self.coord == "px" else "the 0-1000 scale"
            raise BoxError(f"box {[round(v) for v in box]} is empty or too small on {where}")
        return [cx1 / sx, cy1 / sy, cx2 / sx, cy2 / sy]

    def from_unit(self, unit: Box, shown: tuple[int, int]) -> list[int]:
        """Unit box -> integer box in this frame, as the model would write it."""
        sx, sy = self._scale(shown)
        return [round(unit[0] * sx), round(unit[1] * sy), round(unit[2] * sx), round(unit[3] * sy)]


FRAMES = {
    "qwen2.5-vl": Frame("qwen2.5-vl", "px", 28),
    "qwen3": Frame("qwen3", "norm1000", 32),     # Qwen3-VL / Qwen3.5 / Qwen3.8
}


def get_frame(name: str | Frame) -> Frame:
    if isinstance(name, Frame):
        return name
    if name not in FRAMES:
        raise ValueError(f"unknown frame {name!r}; known: {sorted(FRAMES)}")
    return FRAMES[name]


def unit_to_region(unit: Box, region: tuple[float, float, float, float]) -> Box:
    """Unit box of an image that shows `region` (original page pixels) -> original page pixels."""
    left, top, right, bottom = region
    w, h = right - left, bottom - top
    return [float(left + unit[0] * w), float(top + unit[1] * h), float(left + unit[2] * w), float(top + unit[3] * h)]


def crop_region(box: Box, page_size: tuple[int, int], pad: int = CROP_PAD) -> tuple[int, int, int, int]:
    """SCORE crop: a box in original page pixels, padded by `pad` px on each side and clamped to
    the page -> integer crop rectangle (left, top, right, bottom)."""
    w, h = page_size
    left, top = max(math.floor(box[0] - pad), 0), max(math.floor(box[1] - pad), 0)
    right, bottom = min(math.ceil(box[2] + pad), int(w)), min(math.ceil(box[3] + pad), int(h))
    if right <= left or bottom <= top:
        raise BoxError(f"crop box {[round(v) for v in box]} is outside the page")
    if max(right - left, bottom - top) / min(right - left, bottom - top) > MAX_ASPECT:
        raise BoxError(f"crop box {[round(v) for v in box]} has an extreme aspect ratio")
    return left, top, right, bottom
