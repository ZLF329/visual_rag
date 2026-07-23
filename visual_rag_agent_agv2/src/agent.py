from __future__ import annotations

"""AGv2 eval agent: merged-action protocol loop.

Every model turn is:  <think> + [<update_graph> iff observation pending] + one action
(<search>/<bbox>/<answer>).  See src/protocol.py for the parser and DESIGN_AGV2.md for the
full protocol.  This file owns: the eval loop, retrieval (cursor), crop execution, trace/SFT
recording, and run artifacts.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

from src.active_clue_graph import (
    ClueGraph,
    ObservationImage,
    QueryRecord,
    VisualObservation,
    format_graph_state_for_prompt,
    is_sufficient,
    _fact_label,
)
from src.prompts import POLICY_SYSTEM
from src.protocol import (
    BoxFormatError,
    CropContext,
    ProtocolError,
    commit_crop_decision,
    commit_page_decision,
    crop_displayed_box,
    finalize_root,
    finish_crop_chain,
    pending_hint,
)
from src.schemas import GraphDecisionResult


def _rewrite_box_to_displayed_px(raw_text: str, turn: Any, displayed_size: tuple) -> str:
    """Convert a 0-1000 normalized box (Qwen3-VL native frame) to displayed-image pixels,
    mutating turn.box and rewriting the <bbox> payload in the response text so persisted
    targets carry canonical pixel coordinates. Boxes with any coordinate > 1000 are assumed
    to already be pixels and left untouched."""
    if any(v > 1000 for v in turn.box):
        return raw_text
    width, height = displayed_size
    box_px = [
        round(turn.box[0] / 1000.0 * width),
        round(turn.box[1] / 1000.0 * height),
        round(turn.box[2] / 1000.0 * width),
        round(turn.box[3] / 1000.0 * height),
    ]
    turn.box = box_px
    payload = f"<bbox>[{box_px[0]},{box_px[1]},{box_px[2]},{box_px[3]}]</bbox>"
    # Rewrite only after the think block: a <bbox> literal quoted inside <think> must never
    # be the one replaced (the action tag always follows </think>).
    head, sep, tail = raw_text.rpartition("</think>")
    if sep:
        return head + sep + re.sub(r"<bbox>\s*\[[^\]]*\]\s*</bbox>", payload, tail, count=1)
    return re.sub(r"<bbox>\s*\[[^\]]*\]\s*</bbox>", payload, raw_text, count=1)


def is_fatal_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    fatal_markers = (
        "cuda out of memory",
        "cuda error: out of memory",
        "device-side assert",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
    )
    return any(marker in message for marker in fatal_markers)


class Agent:
    def __init__(
        self,
        vlm: Any,
        retriever: Any,
        top_k: int = 1,
        max_iters: int = 5,
        bbox_frame: str = "displayed_px",
    ) -> None:
        self.vlm = vlm
        self.retriever = retriever
        self.top_k = top_k
        self.max_iters = max_iters
        # "displayed_px" (canonical protocol frame) or "norm1000" for teachers that ground in
        # 0-1000 normalized coordinates (Qwen3-VL family) regardless of prompt instructions.
        # With "norm1000" the emitted box is converted to displayed pixels AND rewritten in the
        # recorded response text, so persisted SFT targets stay in the canonical pixel frame.
        self.bbox_frame = bbox_frame

    def run(
        self,
        query: str,
        output_dir: str | Path | None = None,
        deck_name: str | None = None,
    ) -> dict[str, Any]:
        run_dir = Path(output_dir) if output_dir is not None else None
        graph = ClueGraph.from_root_question(query)
        observation: VisualObservation | None = None
        crop_ctx = CropContext()
        trace: list[dict[str, Any]] = []
        seen_page_labels: set[str] = set()
        last_turn: dict[str, Any] | None = None  # {assistant, user, images, image_paths}
        final_answer = ""
        terminated_by = "max_iters"
        step = 0

        while step < self.max_iters:
            pending = observation is not None
            pending_is_crop = pending and observation.kind == "crop"
            messages, images, image_paths = build_turn_messages(
                root_query=query,
                graph=graph,
                observation_pending=pending,
                last_turn=last_turn,
                crop_page_label=crop_ctx.source_page_label if pending_is_crop else "",
                crop_target_node=(crop_ctx.target_node_id or "") if pending_is_crop else "",
            )
            try:
                raw_text, turn = self.vlm.generate_turn(
                    messages=messages,
                    images=images,
                    observation_pending=pending,
                )
            except BoxFormatError as exc:
                # bbox payload malformed on first attempt: soft lane — record and continue with
                # an error observation so the model can retry a well-formed box next turn.
                # We cannot commit the (possibly present) update block since parsing stopped;
                # treat the whole turn as consumed by the box error.
                trace.append({"iter": step, "step": "box_error", "error": str(exc)})
                last_turn = turn_record("(malformed bbox)", f"Invalid bbox: {exc}. "
                                        "Coordinates must be [x1,y1,x2,y2] pixels on the displayed image.")
                step += 1
                continue
            except Exception as exc:
                if is_fatal_model_error(exc):
                    raise
                trace.append({"iter": step, "step": "policy_error", "error": str(exc)})
                terminated_by = "policy_error"
                break

            if self.bbox_frame == "norm1000" and turn.action == "bbox" and turn.box:
                disp = None
                if pending and observation.kind != "crop" and observation.images:
                    disp = observation.images[0].prompt_image().size
                elif crop_ctx.ready:
                    disp = crop_ctx.displayed_size
                if disp is not None:
                    raw_text = _rewrite_box_to_displayed_px(raw_text, turn, disp)

            sft_record = {
                "call_type": "policy",
                "system": messages[0]["content"] if messages else POLICY_SYSTEM,
                "messages": messages,
                "image_paths": image_paths,
                "target": raw_text,
            }
            turn_trace: dict[str, Any] = {
                "iter": step,
                "step": "turn",
                "active_node_id": graph.active_node_id,
                "think": turn.think,
                "action": turn.action,
                "sft": sft_record,
            }

            # ---- 1. commit the pending observation, if any -------------------------------
            commit_error: str | None = None
            commit_echo = ""
            if turn.update_payload is not None:
                try:
                    decision = GraphDecisionResult.model_validate(turn.update_payload)
                except Exception as exc:
                    commit_error = f"invalid update_graph payload: {exc}"
                else:
                    obs_kind = observation.kind if observation is not None else ""
                    obs_summary = observation.summary() if observation is not None else ""
                    try:
                        if obs_kind == "crop":
                            dtype = commit_crop_decision(graph, crop_ctx.target_node_id, decision)
                            turn_trace["commit"] = {"kind": "crop", "type": dtype,
                                                    "target_node": crop_ctx.target_node_id}
                            # crop chain ends unless this turn zooms again: execute the
                            # deferred active move (accept->parent / expand->remaining child).
                            if turn.action != "bbox":
                                finish_crop_chain(graph, crop_ctx)
                        else:
                            # If this turn zooms the page it just committed, keep active
                            # PINNED on the facts node for the crop turn(s); the move the
                            # commit would make is deferred until the chain ends.
                            defer = turn.action == "bbox"
                            dtype, facts_node, resume_id = commit_page_decision(
                                graph, decision, obs_summary, defer_active_shift=defer
                            )
                            turn_trace["commit"] = {"kind": "page", "type": dtype,
                                                    "facts_node": facts_node}
                            page_image = observation.images[0] if observation and observation.images else None
                            if defer and dtype in {"accept", "expand"} and page_image is not None:
                                crop_ctx.source_image = page_image.image
                                crop_ctx.displayed_size = page_image.prompt_image().size
                                crop_ctx.source_page_label = page_image.page_label
                                crop_ctx.target_node_id = facts_node
                                crop_ctx.resume_active_node_id = resume_id
                            else:
                                crop_ctx.clear()
                    except Exception as exc:
                        commit_error = f"update_graph apply failed: {exc}"
                    else:
                        commit_echo = "Graph update:\n" + json.dumps(
                            decision.model_dump(mode="json", exclude_none=True),
                            ensure_ascii=False, separators=(",", ":"),
                        )
                observation = None  # observation is consumed either way

            if commit_error is not None:
                turn_trace["step"] = "update_graph_error"
                turn_trace["error"] = commit_error
                trace.append(turn_trace)
                terminated_by = "update_graph_error"
                break

            turn_trace["graph"] = graph.as_serializable()

            # ---- 2. execute the action ---------------------------------------------------
            if turn.action is None:
                # Commit-only turn: the model defers its action to see the updated graph
                # rendered (typically right before a multi-hop final answer).
                trace.append(turn_trace)
                last_turn = turn_record(raw_text, commit_echo or "Graph updated.")
                step += 1
                continue

            if turn.action == "answer":
                final_answer = turn.action_payload.strip()
                finalize_root(graph, final_answer)
                turn_trace["answer"] = final_answer
                trace.append(turn_trace)
                terminated_by = "answer"
                break

            if turn.action == "search":
                search_query = turn.action_payload.strip()
                active_node = graph.active()
                try:
                    observation = self._search(
                        search_query=search_query,
                        deck_name=deck_name,
                        seen_page_labels=seen_page_labels,
                    )
                except Exception as exc:
                    if is_fatal_model_error(exc):
                        raise
                    observation = VisualObservation(
                        query=search_query, kind="search", message=f"Search failed: {exc}"
                    )
                active_node.query_history.append(QueryRecord(query=search_query, outcome="pending"))
                active_node.num_attempts += 1
                persist_observation_images(
                    observation,
                    run_dir / "observations" / f"iter_{step:03d}_search" if run_dir else None,
                )
                turn_trace["query"] = search_query
                turn_trace["observation"] = observation.as_serializable()
                trace.append(turn_trace)
                last_turn = turn_record(raw_text, format_observation_text(observation),
                                        observation_images(observation),
                                        observation_image_paths(observation))
                step += 1
                continue

            if turn.action == "bbox":
                if not crop_ctx.ready:
                    turn_trace["step"] = "invalid_bbox_no_target"
                    turn_trace["error"] = (
                        "bbox requires a page committed with accept/expand this episode"
                    )
                    trace.append(turn_trace)
                    terminated_by = "policy_error"
                    break
                try:
                    crop_image = crop_displayed_box(
                        crop_ctx.source_image
                        if isinstance(crop_ctx.source_image, Image.Image)
                        else Image.fromarray(crop_ctx.source_image),
                        crop_ctx.displayed_size,
                        turn.box,
                    )
                except BoxFormatError as exc:
                    trace.append({**turn_trace, "step": "box_error", "error": str(exc)})
                    last_turn = turn_record(raw_text, f"Invalid bbox: {exc}.")
                    step += 1
                    continue
                crop_id = f"{make_image_id(crop_ctx.source_page_label, step)}_crop_{step}"
                observation = VisualObservation(
                    query="",
                    kind="crop",
                    images=[
                        ObservationImage(
                            image_id=crop_id,
                            page_label=crop_ctx.source_page_label,
                            image=crop_image,
                            crop_box=list(turn.box),
                        )
                    ],
                )
                persist_observation_images(
                    observation,
                    run_dir / "observations" / f"iter_{step:03d}_crop" if run_dir else None,
                )
                turn_trace["box"] = list(turn.box)
                turn_trace["observation"] = observation.as_serializable()
                trace.append(turn_trace)
                last_turn = turn_record(raw_text, format_observation_text(observation),
                                        observation_images(observation),
                                        observation_image_paths(observation))
                step += 1
                continue

        final = final_answer or graph.root().answer or fallback_answer_from_graph(graph)
        output = {
            "query": query,
            "deck_name": deck_name,
            "answer": final,
            "graph": graph.as_serializable(),
            "trace": trace,
            "terminated_by": terminated_by,
        }
        if run_dir is not None:
            write_run_artifacts(run_dir, output)
        return output

    def _search(
        self,
        *,
        search_query: str,
        deck_name: str | None,
        seen_page_labels: set[str],
    ) -> VisualObservation:
        # cursor: fetch top-K, show only the FIRST unseen page (model sees <=1 image).
        # K via ACTIVE_GRAPH_RETRIEVE_K (default 1 = top-1).
        _k = max(self.top_k, int(os.environ.get("ACTIVE_GRAPH_RETRIEVE_K", "1")))
        try:
            retrieved = self.retriever.search(search_query, top_k=_k, deck_name=deck_name)
        except TypeError:
            retrieved = self.retriever.search(search_query, top_k=_k)

        images: list[ObservationImage] = []
        skipped: list[str] = []
        for idx, (image, page_label) in enumerate(retrieved, start=1):
            label = str(page_label or f"retrieved_{idx}")
            if label in seen_page_labels:
                skipped.append(label)
                continue
            seen_page_labels.add(label)
            images.append(
                ObservationImage(
                    image_id=make_image_id(label, idx),
                    page_label=label,
                    image=image.convert("RGB").copy(),
                )
            )
            break  # cursor: stop at first unseen page

        if not retrieved:
            message = "Search returned no visual candidates."
        elif skipped and not images:
            message = f"Search returned only duplicate pages: {skipped}."
        elif skipped:
            message = f"Skipped duplicate pages: {skipped}."
        else:
            message = ""
        return VisualObservation(query=search_query, images=images, kind="search", message=message)


# ---------------------------------------------------------------------------- helpers


def build_turn_messages(
    *,
    root_query: str,
    graph: ClueGraph,
    observation_pending: bool,
    last_turn: dict[str, Any] | None,
    crop_page_label: str = "",
    crop_target_node: str = "",
) -> tuple[list[dict[str, str]], list[Any], list[str]]:
    """system (+question anchor), the single most recent action-observation pair, then the
    graph block with the pending-obs hint. Mirrors the RL env exactly."""
    system_content = (
        POLICY_SYSTEM
        + "\n\n=== CURRENT TASK - ORIGINAL QUESTION ===\n"
        + str(root_query).strip()
        + "\nKeep this exact question as the goal of every action. Do NOT answer until the"
        " graph holds supporting evidence for EVERY part of it; for multi-part or comparison"
        " questions, gather evidence for each part before answering."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    images: list[Any] = []
    image_paths: list[str] = []
    if last_turn is not None:
        messages.append({"role": "assistant", "content": str(last_turn["assistant"])})
        messages.append({"role": "user", "content": str(last_turn["user"]).strip()})
        images.extend(last_turn.get("images") or [])
        image_paths.extend(str(p) for p in (last_turn.get("image_paths") or []))
    graph_block = (
        "Graph state:\n" + format_graph_state_for_prompt(graph)
        + "\nQuestion to answer right now (active node): " + str(graph.active().question).strip()
        + "\n" + pending_hint(
            observation_pending,
            crop_page_label=crop_page_label,
            crop_target_node=crop_target_node,
        )
    )
    messages.append({"role": "user", "content": graph_block})
    return messages, images, image_paths


def turn_record(
    assistant_text: str,
    user_text: str,
    images: list[Any] | None = None,
    image_paths: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "assistant": str(assistant_text or "").strip(),
        "user": str(user_text or "").strip(),
        "images": list(images or []),
        "image_paths": list(image_paths or []),
    }


def format_observation_text(observation: VisualObservation | None) -> str:
    if observation is None:
        return "Observation: None."
    rows = ["Observation:", f"  kind: {observation.kind}"]
    if observation.kind == "search":
        rows.append(f"  search_query: {observation.query}")
    if observation.message:
        rows.append(f"  note: {observation.message}")
    if not observation.images:
        rows.append("  images: []")
        return "\n".join(rows)
    rows.append("  images:")
    for image in observation.images:
        rows.append(f"  - image_id: {image.image_id}; page_label: {image.page_label}")
        rows.append("    <image>")
    return "\n".join(rows)


def observation_images(observation: VisualObservation | None) -> list[Any]:
    if observation is None:
        return []
    return list(observation.prompt_images())


def observation_image_paths(observation: VisualObservation | None) -> list[str]:
    if observation is None:
        return []
    return [str(image.path) for image in observation.images if image.path]


def persist_observation_images(observation: VisualObservation | None, directory: Path | None) -> None:
    if observation is None or directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(observation.images):
        safe_id = make_image_id(image.image_id, idx + 1)
        path = directory / f"{safe_id}.png"
        image.prompt_image().convert("RGB").save(path)
        image.path = str(path)


def make_image_id(page_label: str, idx: int) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(page_label)).strip("_")
    safe = safe or f"image_{idx}"
    return safe[:96]


def fallback_answer_from_graph(graph: ClueGraph) -> str:
    root = graph.root()
    if root.answer:
        return root.answer
    sufficient_children = [
        graph.nodes[child_id]
        for child_id in root.children
        if is_sufficient(graph.nodes[child_id]) and graph.nodes[child_id].answer
    ]
    if root.known_facts or sufficient_children:
        parts = [_fact_label(f) for f in root.known_facts]
        parts.extend(
            f"{child.question}: {child.answer}"
            for child in sufficient_children
            if child.answer
        )
        return " | ".join(parts)
    return ""


def to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def write_run_artifacts(run_dir: Path, output: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "query.txt").write_text(output["query"], encoding="utf-8")
    (run_dir / "answer.txt").write_text(output.get("answer", ""), encoding="utf-8")
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in output.get("trace", []):
            f.write(json.dumps(row, ensure_ascii=False, default=lambda o: str(o)) + "\n")
    with (run_dir / "graph_final.json").open("w", encoding="utf-8") as f:
        json.dump(output.get("graph", {}), f, ensure_ascii=False, indent=2)
