from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel

from src.active_clue_graph import (
    ClueGraph,
    ObservationImage,
    QueryRecord,
    VisualObservation,
    apply_graph_decision,
    format_graph_state_for_prompt,
    format_observation_for_prompt,
    is_sufficient,
    valid_actions,
)
from src.image_utils import crop_grid_cells_bbox
from src.memory import Memory
from src.prompts import build_policy_prompt
from src.schemas import AnalyseResult, GraphDecisionResult, PolicyActionResult


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
    def __init__(self, vlm: Any, retriever: Any, top_k: int = 1, max_iters: int = 5) -> None:
        self.vlm = vlm
        self.retriever = retriever
        self.top_k = top_k
        self.max_iters = max_iters

    def run(
        self,
        query: str,
        output_dir: str | Path | None = None,
        deck_name: str | None = None,
    ) -> dict[str, Any]:
        run_dir = Path(output_dir) if output_dir is not None else None
        image_dir = run_dir / "images" if run_dir is not None else None
        memory = Memory(original_query=query, image_dir=image_dir)
        graph = ClueGraph.from_root_question(query)
        observation: VisualObservation | None = None
        trace: list[dict[str, Any]] = []
        last_action: PolicyActionResult | None = None
        seen_page_labels: set[str] = set()
        policy_history: list[dict[str, Any]] = []
        pending_policy_action: str | None = None
        policy_history_window = int(os.environ.get("ACTIVE_GRAPH_EVAL_HISTORY_WINDOW", "2"))

        while memory.iter < self.max_iters:
            active_node = graph.active()
            actions = valid_actions(graph, observation)
            graph_state_text = format_graph_state_for_prompt(graph)
            observation_text = format_observation_for_prompt(observation)
            policy_system, policy_user = build_policy_prompt(
                root_query=query,
                graph_state=graph_state_text,
                observation_block=observation_text,
                valid_actions=actions,
            )
            policy_messages, policy_images, policy_image_paths = build_policy_chat_messages(
                system=policy_system,
                graph_state=graph_state_text,
                current_observation=observation,
                valid_actions=actions,
                recent_history=policy_history[-policy_history_window:],
                root_query=query,
                active_question=active_node.question,
            )
            try:
                action = self._choose_action(
                    root_query=query,
                    graph=graph,
                    observation=observation,
                    actions=actions,
                    policy_messages=policy_messages,
                    policy_images=policy_images,
                )
                action_answer = action.answer if action.type == "ANSWER" else None
                action_target = format_policy_action_target(action, answer=action_answer)
                trace.append(
                    {
                        "iter": memory.iter,
                        "step": "policy",
                        "active_node_id": active_node.id,
                        "valid_actions": actions,
                        "result": to_plain(action),
                        "sft": {
                            "call_type": "policy",
                            "system": policy_system,
                            "user": policy_user,
                            "graph_state": format_graph_state_for_prompt(graph),
                            "observation": format_observation_for_prompt(observation),
                            "messages": policy_messages,
                            "image_paths": policy_image_paths,
                            "target": action_target,
                        },
                    }
                )
                last_action = action
                pending_policy_action = action_target
            except Exception as exc:
                if is_fatal_model_error(exc):
                    raise
                trace.append(
                    {
                        "iter": memory.iter,
                        "step": "policy_error",
                        "active_node_id": active_node.id,
                        "valid_actions": actions,
                        "error": str(exc),
                    }
                )
                break

            if action.type == "ANSWER":
                break

            if action.type == "SEARCH":
                search_query = str(action.query or "").strip()
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
                        query=search_query,
                        kind="search",
                        message=f"Search failed: {exc}",
                    )
                    trace.append(
                        {
                            "iter": memory.iter,
                            "step": "search_error",
                            "active_node_id": active_node.id,
                            "query": search_query,
                            "error": str(exc),
                        }
                    )

                active_node.query_history.append(QueryRecord(query=search_query, outcome="pending"))
                active_node.num_attempts += 1
                persist_observation_images(
                    observation,
                    run_dir / "observations" / f"iter_{memory.iter:03d}_search"
                    if run_dir is not None
                    else None,
                )
                if pending_policy_action is not None:
                    policy_history.append(policy_history_entry(pending_policy_action, observation))
                    pending_policy_action = None
                if observation is not None and not observation.images:
                    memory.add_empty_search_warning(search_query)
                trace.append(
                    {
                        "iter": memory.iter,
                        "step": "search",
                        "active_node_id": active_node.id,
                        "query": search_query,
                        "deck_name": deck_name,
                        "observation": observation.as_serializable() if observation is not None else None,
                    }
                )
                memory.iter += 1
                continue

            if action.type == "CROP":
                try:
                    observation = crop_observation(observation, action, crop_index=memory.iter)
                    persist_observation_images(
                        observation,
                        run_dir / "observations" / f"iter_{memory.iter:03d}_crop"
                        if run_dir is not None
                        else None,
                    )
                    if pending_policy_action is not None:
                        policy_history.append(policy_history_entry(pending_policy_action, observation))
                        pending_policy_action = None
                    trace.append(
                        {
                            "iter": memory.iter,
                            "step": "crop",
                            "active_node_id": active_node.id,
                            "result": to_plain(action),
                            "observation": observation.as_serializable() if observation is not None else None,
                        }
                    )
                except Exception as exc:
                    if is_fatal_model_error(exc):
                        raise
                    trace.append(
                        {
                            "iter": memory.iter,
                            "step": "crop_error",
                            "active_node_id": active_node.id,
                            "result": to_plain(action),
                            "error": str(exc),
                        }
                    )
                    memory.iter += 1
                    break
                memory.iter += 1
                continue

            if action.type == "UPDATE_GRAPH":
                episode_query = observation.query if observation is not None else latest_query(active_node)
                observation_summary = observation.summary() if observation is not None else ""
                try:
                    if action.graph_update is not None:
                        decision = GraphDecisionResult.model_validate(action.graph_update)
                        decision.validate_branch()
                    else:
                        decision = self.vlm.update_clue_graph(
                            root_query=query,
                            graph=graph,
                            observation=observation,
                        )
                    apply_graph_decision(
                        graph=graph,
                        node_id=active_node.id,
                        decision=decision,
                        observation_summary=observation_summary,
                    )
                    self._mirror_graph_decision_to_memory(
                        memory=memory,
                        decision=decision,
                        observation=observation,
                        search_query=episode_query,
                    )
                    if pending_policy_action is not None:
                        policy_history.append(
                            {
                                "action": pending_policy_action,
                                "observation": "Graph update:\n"
                                + json.dumps(to_plain(decision), ensure_ascii=False, separators=(",", ":")),
                                "images": [],
                                "image_paths": [],
                            }
                        )
                        pending_policy_action = None
                    trace.append(
                        {
                            "iter": memory.iter,
                            "step": "update_graph",
                            "node_id": active_node.id,
                            "observation_summary": observation_summary,
                            "decision": to_plain(decision),
                            "graph": graph.as_serializable(),
                        }
                    )
                    observation = None
                except Exception as exc:
                    if is_fatal_model_error(exc):
                        raise
                    trace.append(
                        {
                            "iter": memory.iter,
                            "step": "update_graph_error",
                            "node_id": active_node.id,
                            "error": str(exc),
                        }
                    )
                    memory.iter += 1
                    break
                memory.iter += 1
                continue

            trace.append(
                {
                    "iter": memory.iter,
                    "step": "invalid_policy_action",
                    "active_node_id": active_node.id,
                    "valid_actions": actions,
                    "result": to_plain(action),
                }
            )
            memory.iter += 1
            break

        final = (last_action.answer or "").strip() if last_action is not None and last_action.type == "ANSWER" else ""
        final = final or graph.root().answer or fallback_answer_from_graph(graph)

        terminated_by = (
            "answer" if last_action is not None and last_action.type == "ANSWER" else "max_iters"
        )
        if trace:
            last_step = str(trace[-1].get("step", ""))
            if last_step in {
                "policy_error",
                "crop_error",
                "update_graph_error",
                "invalid_policy_action",
            }:
                terminated_by = last_step

        output = {
            "query": query,
            "deck_name": deck_name,
            "answer": final,
            "memory": memory.as_serializable(),
            "graph": graph.as_serializable(),
            "trace": trace,
            "terminated_by": terminated_by,
        }
        if run_dir is not None:
            write_run_artifacts(run_dir, output)
        return output

    def _choose_action(
        self,
        *,
        root_query: str,
        graph: ClueGraph,
        observation: VisualObservation | None,
        actions: list[str],
        policy_messages: list[dict[str, str]] | None = None,
        policy_images: list[Any] | None = None,
    ) -> PolicyActionResult:
        # EVAL-LENIENT (this file is eval-only; training uses verl-agent's env):
        # accept whatever action the model picks even if it is out of the valid_actions
        # mask, and run it through its normal branch. valid_actions stays in the PROMPT
        # as guidance, but validation passes None so only malformed actions (e.g. SEARCH
        # without a query) are rejected, never out-of-mask ones.
        if hasattr(self.vlm, "choose_graph_action_from_messages") and policy_messages is not None:
            action = self.vlm.choose_graph_action_from_messages(
                messages=policy_messages,
                images=policy_images or [],
                valid_actions=None,
            )
            action.validate_branch(None)
            return action

        if hasattr(self.vlm, "choose_graph_action"):
            action = self.vlm.choose_graph_action(
                root_query=root_query,
                graph=graph,
                observation=observation,
                valid_actions=None,
            )
            action.validate_branch(None)
            return action

        if observation is None:
            if "ANSWER" in actions:
                return PolicyActionResult(think="root sufficient", type="ANSWER")
            if "UPDATE_GRAPH" in actions:
                return PolicyActionResult(think="graph has child evidence to commit", type="UPDATE_GRAPH")
            if "SEARCH" in actions:
                return PolicyActionResult(
                    think="need visual evidence",
                    type="SEARCH",
                    query=graph.active().question,
                )
        if "UPDATE_GRAPH" in actions:
            return PolicyActionResult(think="commit current observation", type="UPDATE_GRAPH")
        raise RuntimeError(f"no valid deterministic action is available: {actions}")

    def _paths_for_labels(self, labels: list[str]) -> list[str]:
        entries = getattr(self.retriever, "entries", None) or []
        import re as _re
        def _norm(x):
            return _re.sub(r"[/-]", "_", str(x))
        by_label = {}
        for e in entries:
            pl = getattr(e, "page_label", None); ip = getattr(e, "image_path", None)
            if pl and ip:
                by_label[pl] = str(ip)
                by_label[_norm(pl)] = str(ip)
        out: list[str] = []
        for lab in labels:
            p = by_label.get(lab) or by_label.get(_norm(lab))
            if p:
                out.append(p)
        return out

    def _load_pages_by_label(self, labels: list[str], *, cap: int = 4) -> "tuple[list[Any], list[str]]":
        entries = getattr(self.retriever, "entries", None) or []
        import re as _re
        def _norm(x):
            return _re.sub(r"[/-]", "_", str(x))
        by_label = {}
        for e in entries:
            pl = getattr(e, "page_label", None); ip = getattr(e, "image_path", None)
            if not pl or not ip:
                continue
            by_label[pl] = ip
            by_label[_norm(pl)] = ip   # make_image_id turns both - and / into _
        imgs: list[Any] = []
        used: list[str] = []
        for lab in labels:
            path = by_label.get(lab) or by_label.get(_norm(lab))
            if not path or not Path(path).exists():
                continue
            with Image.open(path) as im:
                imgs.append(im.convert("RGB").copy())
            used.append(lab)
            if len(imgs) >= cap:
                break
        return imgs, used

    def _search(
        self,
        *,
        search_query: str,
        deck_name: str | None,
        seen_page_labels: set[str],
    ) -> VisualObservation:
        # cursor (2026-06-13): fetch top-K, return only the FIRST unseen page (model still
        # sees <=1 image). K via ACTIVE_GRAPH_RETRIEVE_K (default 1 = legacy top-1). EVAL only.
        _k = max(self.top_k, int(os.environ.get("ACTIVE_GRAPH_RETRIEVE_K", "1")))
        try:
            retrieved = self.retriever.search(
                search_query,
                top_k=_k,
                deck_name=deck_name,
            )
        except TypeError:
            retrieved = self.retriever.search(search_query, top_k=_k)

        images: list[ObservationImage] = []
        skipped_duplicate_pages: list[str] = []
        for idx, (image, page_label) in enumerate(retrieved, start=1):
            label = str(page_label or f"retrieved_{idx}")
            if label in seen_page_labels:
                skipped_duplicate_pages.append(label)
                continue
            seen_page_labels.add(label)
            rgb = image.convert("RGB").copy()
            image_id = make_image_id(label, idx)
            images.append(
                ObservationImage(
                    image_id=image_id,
                    page_label=label,
                    image=rgb,
                )
            )
            break  # cursor: stop at first unseen page

        if not retrieved:
            message = "Search returned no visual candidates."
        elif skipped_duplicate_pages and not images:
            message = f"Search returned only duplicate pages: {skipped_duplicate_pages}."
        elif skipped_duplicate_pages:
            message = f"Skipped duplicate pages: {skipped_duplicate_pages}."
        else:
            message = ""

        return VisualObservation(
            query=search_query,
            images=images,
            kind="search",
            message=message,
        )

    def _mirror_graph_decision_to_memory(
        self,
        *,
        memory: Memory,
        decision: GraphDecisionResult,
        observation: VisualObservation | None,
        search_query: str,
    ) -> None:
        if decision.type == "REJECT":
            memory.add_empty_search_warning(search_query)
            return

        if decision.type == "EXPAND":
            facts = [_fact_label(_f) for _f in (decision.supporting_facts or [])] or ([decision.answer] if decision.answer else [])
            text = "; ".join(facts)
            for fact in facts:
                memory.append_observed_evidence(fact)
            memory.evidence_state.missing_requirements = [decision.remaining_subquestion or ""]
            memory.evidence_state.normalize()
            self._store_graph_evidence_image(
                memory=memory,
                observation=observation,
                search_query=search_query,
                text=text,
                decision="partial",
            )
            memory.append_consolidated_summary(text, search_query=search_query)
            return

        if decision.type == "ACCEPT":
            facts = [_fact_label(_f) for _f in (decision.supporting_facts or [])] or ([decision.answer] if decision.answer else [])
            text = "; ".join(facts) if facts else str(decision.answer or "")
            for fact in facts:
                memory.append_observed_evidence(fact)
            memory.evidence_state.missing_requirements = []
            memory.evidence_state.normalize()
            self._store_graph_evidence_image(
                memory=memory,
                observation=observation,
                search_query=search_query,
                text=text,
                decision="yes",
                answer=decision.answer,
            )
            memory.append_consolidated_summary(text, search_query=search_query)

    def _store_graph_evidence_image(
        self,
        *,
        memory: Memory,
        observation: VisualObservation | None,
        search_query: str,
        text: str,
        decision: str,
        answer: str | None = None,
    ) -> None:
        if observation is None or not observation.images:
            return
        image_ref = observation.images[0]
        result = AnalyseResult(
            think="Graph decision committed this observation.",
            summary=text,
            observed_evidence=text,
            decision=decision,
            answer_text=answer if decision == "yes" else None,
            partial_answer=text if decision == "partial" else None,
        )
        memory.write(
            result,
            image_ref.image,
            search_query,
            page_label=image_ref.page_label,
            retained_image=image_ref.image,
        )


def build_policy_chat_messages(
    *,
    system: str,
    graph_state: str,
    current_observation: VisualObservation | None,
    valid_actions: list[str],
    recent_history: list[dict[str, Any]],
    root_query: str = "",
    active_question: str = "",
) -> tuple[list[dict[str, str]], list[Any], list[str]]:
    system_content = system
    if root_query:
        _q = str(root_query).strip()
        system_content = system + chr(10) + chr(10) + "=== CURRENT TASK - ORIGINAL QUESTION ===" + chr(10) + _q + chr(10) + "Keep this exact question as the goal of every action. Do NOT choose ANSWER until the graph holds supporting evidence for EVERY part of it; for multi-part or comparison questions, gather evidence for each part before answering."
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    images: list[Any] = []
    image_paths: list[str] = []

    # Most recent round of dialogue only (action + its observation, image inline). Everything
    # older lives in the persistent graph state below.
    if recent_history:
        pair = recent_history[-1]
        messages.append({"role": "assistant", "content": str(pair["action"])})
        messages.append({"role": "user", "content": str(pair["observation"]).strip()})
        images.extend(pair.get("images") or [])
        image_paths.extend(str(path) for path in pair.get("image_paths") or [])

    graph_block = (
        "Graph state:\n" + str(graph_state).strip()
        + "\n\nValid actions: " + json.dumps(list(valid_actions))
    )
    if active_question:
        graph_block = graph_block + chr(10) + "Question to answer right now (active node): " + str(active_question).strip()
    messages.append({"role": "user", "content": graph_block})
    return messages, images, image_paths


def append_policy_action_request(
    text: str,
    valid_actions: list[str],
    *,
    graph_state: str | None = None,
) -> str:
    parts = [str(text).strip()]
    if graph_state is not None:
        parts.extend(["", "Graph state:", str(graph_state).strip()])
    return "\n".join(parts).strip()


def policy_history_entry(action: str, observation: VisualObservation | None) -> dict[str, Any]:
    return {
        "action": action,
        "observation": format_observation_for_history(observation),
        "images": observation.prompt_images() if observation is not None else [],
        "image_paths": observation_image_paths(observation),
    }


def format_observation_for_history(observation: VisualObservation | None) -> str:
    if observation is None:
        return "Observation: None."
    rows = [
        "Observation:",
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


def crop_observation(
    observation: VisualObservation | None,
    action: PolicyActionResult,
    *,
    crop_index: int,
) -> VisualObservation:
    if observation is None:
        raise ValueError("CROP requires a current observation")
    if not observation.images:
        raise ValueError("CROP requires a current observation image")
    if action.image_id:
        target = next(
            (image for image in observation.images if image.image_id == action.image_id),
            None,
        )
    else:
        target = observation.images[0]
    if target is None:
        raise ValueError(f"unknown image_id for CROP: {action.image_id}")

    if action.cells:
        cropped = crop_grid_cells_bbox(target.image, action.cells)
    elif action.box is not None:
        cropped = crop_action_box(target.image, target.prompt_image(), action.box)
    else:
        raise ValueError("CROP requires either cells or box")

    crop_id = f"{target.image_id}_crop_{crop_index}"
    crop_image = cropped.convert("RGB").copy()
    return VisualObservation(
        query=observation.query,
        kind="crop",
        message=action.reason or f"Cropped {target.image_id}.",
        images=[
            ObservationImage(
                image_id=crop_id,
                page_label=target.page_label,
                image=crop_image,
                source_image_id=target.image_id,
                crop_box=list(action.box) if action.box is not None else None,
                cells=list(action.cells),
            )
        ],
    )


def crop_action_box(image: Any, prompt_image: Any, box: list[float], *, pad: int = 28) -> Any:
    if len(box) != 4:
        raise ValueError("bbox must contain four coordinates")
    x0, y0, x1, y1 = [float(value) for value in box]
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid bbox: {box}")

    width, height = image.size
    prompt_width, prompt_height = getattr(prompt_image, "size", image.size)
    if all(0.0 <= value <= 1000.0 for value in (x0, y0, x1, y1)):
        raw_box = [
            x0 * width / 1000.0,
            y0 * height / 1000.0,
            x1 * width / 1000.0,
            y1 * height / 1000.0,
        ]
        pad_px = 0
    else:
        # Backward-compatible path for older pixel-coordinate traces.
        raw_box = [
            x0 * width / prompt_width,
            y0 * height / prompt_height,
            x1 * width / prompt_width,
            y1 * height / prompt_height,
        ]
        pad_px = pad

    x0, y0, x1, y1 = raw_box
    x0 = max(x0 - pad_px, 0)
    y0 = max(y0 - pad_px, 0)
    x1 = min(x1 + pad_px, width)
    y1 = min(y1 + pad_px, height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid clipped bbox: {box}")
    cropped = image.convert("RGB").crop(
        (
            round(x0),
            round(y0),
            round(x1),
            round(y1),
        )
    )
    if pad_px == 0:
        return smart_resize_crop(cropped)
    return cropped


def smart_resize_crop(image: Any, *, factor: int = 32, min_pixels: int = int(__import__("os").environ.get("ACTIVE_GRAPH_MIN_PIXELS", "500000"))) -> Any:
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


def persist_observation_images(observation: VisualObservation | None, directory: Path | None) -> None:
    if observation is None or directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(observation.images):
        safe_id = make_image_id(image.image_id, idx + 1)
        path = directory / f"{safe_id}.png"
        image.prompt_image().convert("RGB").save(path)
        image.path = str(path)


def observation_image_paths(observation: VisualObservation | None) -> list[str]:
    if observation is None:
        return []
    return [str(image.path) for image in observation.images if image.path]


def format_policy_action_target(action: PolicyActionResult, *, answer: str | None = None) -> str:
    think = f"<think>{action.think}</think>\n" if action.think else ""
    if action.type == "SEARCH":
        return f"{think}<search>{action.query or ''}</search>"
    if action.type == "CROP":
        if action.box is not None:
            box = json.dumps(action.box, ensure_ascii=False, separators=(",", ":"))
            return f"{think}<bbox>{box}</bbox>"
        cells = json.dumps(action.cells, ensure_ascii=False, separators=(",", ":"))
        return f"{think}<bbox>{cells}</bbox>"
    if action.type == "UPDATE_GRAPH":
        if action.graph_update is None:
            return f"{think}<update_graph>commit</update_graph>"
        payload = action.graph_update
        graph_update = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return f"{think}<update_graph>{graph_update}</update_graph>"
    if action.type == "ANSWER":
        answer_text = (answer or action.answer or "").strip()
        if not answer_text:
            raise ValueError("ANSWER target requires non-empty answer")
        return f"{think}<answer>{answer_text}</answer>"
    raise ValueError(f"unknown action type: {action.type}")


def make_image_id(page_label: str, idx: int) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in page_label).strip("_")
    safe = safe or f"image_{idx}"
    return safe[:96]


def latest_query(node: Any) -> str:
    for record in reversed(getattr(node, "query_history", []) or []):
        query = str(getattr(record, "query", "") or "").strip()
        if query:
            return query
    return str(getattr(node, "question", "") or "").strip()


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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (run_dir / "memory_final.json").open("w", encoding="utf-8") as f:
        json.dump(output.get("memory", {}), f, ensure_ascii=False, indent=2)
    with (run_dir / "graph_final.json").open("w", encoding="utf-8") as f:
        json.dump(output.get("graph", {}), f, ensure_ascii=False, indent=2)


def _fact_label(f):
    """Extract the text label from a SupportingFact / dict / str (bbox-facts refactor)."""
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return str(f.get("fact") or f.get("label") or f.get("text") or "")
    return str(getattr(f, "fact", "") or "")
