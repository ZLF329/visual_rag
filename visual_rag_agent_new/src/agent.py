from __future__ import annotations

"""new eval agent: one loop of (page edit, action) turns ending in answer (DESIGN.md §3-§6, §11.8).
Owns retrieval (cursor over unseen pages), revisit, zoom execution, the hard turn budget,
trace / per-turn SFT records and run artifacts. Protocol, memory and context construction live
in src/protocol.py, src/memory.py and src/prompts.py and are shared with rl/envs.py.

Each turn's page edit records what the page in view gives; its tree edit (DESIGN.md §11.23) concludes
from the page memory and attaches the pages it lists; the final evidence pages are the pages attached to
the question tree.

Coordinate frames (DESIGN.md §11.3): the model runs in its native `frame`; each SFT row is rendered
in `sft_frame` (the student's), with every box converted through original page pixels.
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from src.memory import Memory, PageNode, View, Visit, Zoom, apply_edit, resolve_fact_id
from src.prompts import build_turn_messages, first_search_response, format_retry_suffix, turn_of, with_retry_suffix
from src.protocol import (
    Act,
    Box,
    EditError,
    Frame,
    ProtocolError,
    Turn,
    crop_region,
    get_frame,
    parse_turn,
    render_turn,
    unit_to_region,
)

# Experiment switches (DESIGN.md §11.5): who makes the first search, and the guided search wording.
FIRST_SEARCH = ("model", "question")

NO_NEW_PAGE_NOTE = "Search returned no new page (every candidate was already seen). Use a different query, revisit a page, or answer."


def is_fatal_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in (
        "cuda out of memory", "cuda error: out of memory", "device-side assert",
        "cublas_status_alloc_failed", "cudnn_status_alloc_failed",
    ))


def prepare_action(m: Memory, view: View | None, act: Act, frame: Frame, *, turn: int) -> Zoom | None:
    """Validate the action against the (edited) memory before anything is committed; build the
    zoom for crop (SCORE: box -> original page pixels, +28 px, clamp, crop the original).
    Raises EditError (soft lane). Shared with rl/envs.py."""
    action = act.action
    if action.type == "crop":
        page = m.pages[view.page]
        if action.fact_id is not None:
            fact = page.fact(resolve_fact_id(action.fact_id, page.id))
            if fact is None:
                raise EditError(f"crop: {action.fact_id} is not a fact of the page in view ({page.id})")
            box = fact.bbox
        else:
            w, h = page.image.size
            box = unit_to_region(frame.to_unit(action.bbox, frame.shown_size((w, h))), (0, 0, w, h))
        region = crop_region(box, page.image.size)
        return Zoom(image=page.image.crop(region), region=region, turn=turn)
    if action.type == "revisit" and action.page_id not in m.pages:
        raise EditError(f"revisit: unknown page {action.page_id}; pages seen: {', '.join(m.pages) or 'none'}")
    return None


def reframe_box(box: Box, size: tuple[int, int], src: Frame, dst: Frame) -> list[int]:
    """A box on an image of original `size`, from frame `src` to frame `dst`."""
    return dst.from_unit(src.to_unit(box, src.shown_size(size)), dst.shown_size(size))


def reframe_act(act: Act, m: Memory, view: View | None, src: Frame, dst: Frame) -> Act:
    """Every box of a validated act, rewritten from frame `src` to frame `dst` (same images)."""
    out = act.model_copy(deep=True)
    if view is None:
        return out
    page_size = m.pages[view.page].image.size
    ops_size = view.zoom.image.size if view.zoom is not None else page_size
    for op in out.page.facts if out.page is not None else []:
        op.bbox = reframe_box(op.bbox, ops_size, src, dst)
    if out.action.type == "crop" and out.action.bbox is not None:
        out.action.bbox = reframe_box(out.action.bbox, page_size, src, dst)
    return out


class Agent:
    def __init__(
        self,
        vlm: Any,
        retriever: Any,
        *,
        max_turns: int = 10,
        retrieve_k: int | None = None,
        frame: str | Frame = "qwen2.5-vl",
        sft_frame: str | Frame | None = None,
        first_search: str = "model",
        search_guidance: bool = False,
        teacher_hint: str = "",
        format_retries: int = 0,
        image_max_pixels: int | None = None,
    ) -> None:
        """vlm.generate(messages, images) -> str; retriever.search(query, top_k, deck_name) ->
        [(PIL page, label)]. `frame`: the running model's native coordinate frame; `sft_frame`:
        the frame SFT rows are rendered in (default: the same). first_search="question": turn 0 is a
        fixed search with the question itself (no model call, no SFT row) and counts against the
        budget. search_guidance: the guided search wording in the system prompt. teacher_hint: a note
        appended to the question the teacher sees, never to the SFT rows (DESIGN.md §11.15). image_max_pixels:
        a smaller image budget for the running model only; SFT rows keep sft_frame's images (§11.25)."""
        if first_search not in FIRST_SEARCH:
            raise ValueError(f"first_search must be one of {FIRST_SEARCH}, got {first_search!r}")
        if max_turns <= int(first_search == "question"):
            raise ValueError("max_turns must leave at least one model turn after the automatic search")
        self.vlm = vlm
        self.retriever = retriever
        self.max_turns = max_turns
        self.retrieve_k = retrieve_k or int(os.environ.get("ACTIVE_GRAPH_RETRIEVE_K", "5"))
        self.frame = get_frame(frame)
        self.sft_frame = get_frame(sft_frame) if sft_frame is not None else self.frame
        if image_max_pixels:
            self.frame = self.frame.with_max_pixels(int(image_max_pixels))
        self.first_search = first_search
        self.search_guidance = search_guidance
        self.teacher_hint = (teacher_hint or "").strip()
        self.format_retries = format_retries

    # ------------------------------------------------------------------ episode

    def run(self, query: str, output_dir: str | Path | None = None, deck_name: str | None = None) -> dict[str, Any]:
        run_dir = Path(output_dir) if output_dir is not None else None
        m = Memory(query)
        view: View | None = None
        last_response: str | None = None      # in the model's frame
        last_sft: str | None = None           # the same response in the SFT frame
        note = ""
        trace: list[dict[str, Any]] = []
        counts: Counter = Counter()
        answer = ""
        terminated_by = "budget_timeout"
        turn = 0
        if self.first_search == "question":
            page, note = self._search(m, query, deck_name, turn)
            counts["search"] += 1
            view = View(page=page.id, via="search", query=query) if page is not None else None
            trace.append({"turn": turn, "step": "auto_search", "query": query, "retrieved": page.label if page else None})
            last_response = last_sft = first_search_response(query)
            turn += 1

        while turn < self.max_turns:
            force = False
            kind = "none" if view is None else view.kind
            ctx = {"search_guidance": self.search_guidance, "note": note, "force": force,
                   "turn_of": turn_of(turn, self.max_turns, self.first_search)}
            messages, images = build_turn_messages(m, view, last_response, frame=self.frame,
                                                   teacher_hint=self.teacher_hint, **ctx)
            sft_messages = messages if self.sft_frame == self.frame and not self.teacher_hint else build_turn_messages(
                m, view, last_sft, frame=self.sft_frame, **ctx)[0]
            sft_paths = self._shown_paths(m, view, run_dir) if images else []
            note = ""
            raw, parsed, error = self._generate(messages, images, kind, view.page if view else None, force)
            if parsed is None:
                trace.append({"turn": turn, "step": "format_error", "view": kind, "error": error, "raw": raw})
                terminated_by = "format_error"
                break
            act = parsed.act
            target = render_turn(parsed.think, act)
            row: dict[str, Any] = {
                "turn": turn, "step": "turn", "view": kind, "page": view.page if view else None, "forced": force,
                "answer_only": False, "think": parsed.think, "act": act.to_payload(), "raw": raw,
                "sft": {"messages": sft_messages, "image_paths": sft_paths, "target": target},
            }
            try:
                new_m, events = apply_edit(m, act.page, act.tree, view, turn=turn, frame=self.frame)
                zoom = prepare_action(new_m, view, act, self.frame, turn=turn)
            except EditError as exc:
                row["step"], row["error"] = "edit_error", str(exc)
                trace.append(row)
                note = f"your previous response was NOT applied ({exc}). Fix it and respond again."
                last_response = last_sft = target    # a trajectory with an error turn is never kept for SFT
                turn += 1
                continue
            if self.sft_frame != self.frame:
                row["sft"]["target"] = render_turn(parsed.think, reframe_act(act, m, view, self.frame, self.sft_frame))
            m = new_m
            row["events"] = events
            action = act.action

            if action.type == "answer":
                answer = action.text
                m.finish(answer)
                terminated_by = "answer"
                row["memory"] = m.as_serializable()
                trace.append(row)
                break
            counts[action.type] += 1
            if turn + 1 >= self.max_turns:
                row["memory"] = m.as_serializable()
                trace.append(row)
                turn += 1
                break
            if action.type == "crop":
                view.zoom = zoom
            elif action.type == "revisit":
                m.pages[action.page_id].visits.append(Visit(turn, m.active_id, "revisit"))
                view = View(page=action.page_id, via="revisit")
            else:
                page, note = self._search(m, action.query, deck_name, turn)
                view = View(page=page.id, via="search", query=action.query) if page is not None else None
                if page is not None:
                    row["retrieved"] = page.label
            row["memory"] = m.as_serializable()
            trace.append(row)
            last_response, last_sft = target, row["sft"]["target"]
            turn += 1

        output = {
            "query": query,
            "deck_name": deck_name,
            "answer": answer,
            "final_pages": [m.pages[p].label for p in m.evidence_pages()],
            "memory": m.as_serializable(),
            "counts": dict(counts),
            "turns": min(turn + 1, self.max_turns),
            "trace": trace,
            "terminated_by": terminated_by,
        }
        if run_dir is not None:
            write_run_artifacts(run_dir, output)
        return output

    # ------------------------------------------------------------------ generation

    def _generate(self, messages, images, kind, current_page, force) -> tuple[str, Turn | None, str | None]:
        raw = self.vlm.generate(messages, images)
        error: str | None = None
        retries = 0 if force else self.format_retries
        for attempt in range(retries + 1):
            if attempt > 0:
                raw = self.vlm.generate(with_retry_suffix(messages, format_retry_suffix(error)), images)
            try:
                return raw, parse_turn(raw, view=kind, current_page=current_page, force_answer=force), None
            except ProtocolError as exc:
                error = str(exc)
        return raw, None, error

    def _shown_paths(self, m: Memory, view: View, run_dir: Path | None) -> list[str]:
        """The images in view as the SFT frame shows them, saved once per page / zoom."""
        items = [(view.page, m.pages[view.page].image)]
        if view.zoom is not None:
            items.append((f"{view.page}_zoom_t{view.zoom.turn:03d}", view.zoom.image))
        return self._save_shown(items, run_dir)

    def _save_shown(self, items: list[tuple[str, Image.Image]], run_dir: Path | None) -> list[str]:
        if run_dir is None:
            return ["" for _ in items]
        directory = run_dir / "observations"
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for key, image in items:
            path = directory / f"{key}.{self.sft_frame.name}.png"
            if not path.exists():
                self.sft_frame.show(image).save(path)
            paths.append(str(path))
        return paths

    # ------------------------------------------------------------------ retrieval

    def _search(self, m: Memory, query: str, deck_name: str | None, turn: int
                ) -> tuple[PageNode | None, str]:
        """Cursor retrieval: top-k, the first page not seen in this episode becomes a new page node."""
        try:
            retrieved = self.retriever.search(query, top_k=self.retrieve_k, deck_name=deck_name)
        except TypeError:
            retrieved = self.retriever.search(query, top_k=self.retrieve_k)
        for image, label in retrieved:
            label = str(label)
            if label in m.aliases:
                continue
            page = m.add_page(label, image.convert("RGB").copy(), query)
            page.visits.append(Visit(turn, m.active_id, "search", query))
            return page, ""
        return None, NO_NEW_PAGE_NOTE


# ---------------------------------------------------------------------- helpers


def write_run_artifacts(run_dir: Path, output: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "query.txt").write_text(output["query"], encoding="utf-8")
    (run_dir / "answer.txt").write_text(output.get("answer", ""), encoding="utf-8")
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in output.get("trace", []):
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with (run_dir / "memory_final.json").open("w", encoding="utf-8") as f:
        json.dump(output.get("memory", {}), f, ensure_ascii=False, indent=2)
