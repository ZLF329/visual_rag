from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
from PIL import Image

from agent_system.environments.prompts import SLIDEVQA_ACTIVE_GRAPH_TEMPLATE, SLIDEVQA_TEMPLATE

_DEFAULT_VISUAL_RAG_AGENT_PATH = Path("/scratch/punim0614/lifuzhang/visual_rag_agent")
if _DEFAULT_VISUAL_RAG_AGENT_PATH.exists() and str(_DEFAULT_VISUAL_RAG_AGENT_PATH) not in sys.path:
    sys.path.insert(0, str(_DEFAULT_VISUAL_RAG_AGENT_PATH))

try:
    from src.active_clue_graph import (
        ClueGraph,
        ObservationImage,
        QueryRecord,
        VisualObservation,
        apply_graph_decision,
        format_graph_state_for_prompt,
        format_observation_for_prompt,
        is_sufficient,
    )
    from src.schemas import GraphDecisionResult
    from src import protocol as agv2
except Exception:
    ClueGraph = None
    ObservationImage = None
    QueryRecord = None
    VisualObservation = None
    GraphDecisionResult = None
    apply_graph_decision = None
    format_graph_state_for_prompt = None
    format_observation_for_prompt = None
    is_sufficient = None
    agv2 = None

def _load_active_graph_policy_system() -> str:
    # 1) normal import (works wherever `src` is importable, e.g. the driver with PYTHONPATH set).
    try:
        from src.prompts import POLICY_SYSTEM as policy_system
        if policy_system:
            return str(policy_system)
    except Exception:
        pass
    # 2) Ray rollout workers do NOT inherit the launcher PYTHONPATH, so the bare import above
    # fails in them. Load eval's prompts.py directly from a known visual_rag_agent location so
    # the RL policy prompt stays IDENTICAL to eval's bbox-format POLICY_SYSTEM. Try the
    # env-configured path first, then the AutoDL/Spartan real paths -- do NOT rely on a symlink
    # (the /scratch symlink proved unreliable inside workers and silently fell through to the
    # stale strings-format template below -> model emitted bbox-less facts -> 1.9% success).
    candidates = []
    _env_vra = os.environ.get("VISUAL_RAG_AGENT_PATH")
    if _env_vra:
        candidates.append(Path(_env_vra) / "src" / "prompts.py")
    candidates.append(Path("/root/autodl-tmp/visual_rag_agent/src/prompts.py"))
    candidates.append(Path("/scratch/punim0614/lifuzhang/visual_rag_agent/src/prompts.py"))
    for prompt_path in candidates:
        try:
            if not prompt_path.exists():
                continue
            spec = importlib.util.spec_from_file_location("active_graph_eval_prompts", prompt_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                policy_system = getattr(module, "POLICY_SYSTEM", None)
                if policy_system:
                    return str(policy_system)
        except Exception:
            continue
    return SLIDEVQA_ACTIVE_GRAPH_TEMPLATE.split('Root question:')[0].strip()


ACTIVE_GRAPH_POLICY_SYSTEM = _load_active_graph_policy_system()


@dataclass
class SlideVQAState:
    question: str
    answer: str
    deck_name: str | None
    data_source: str
    reference_pages: set[str]
    max_steps: int
    sample_id: str = ""
    mode: str = "ra"
    active_question: str = ""
    graph: Any | None = None
    observation: Any | None = None
    seen_page_labels: set[str] = field(default_factory=set)
    iter: int = 0
    done: bool = False
    final_answer: str = ""
    retrieved_page_history: list[str] = field(default_factory=list)
    hit_pages: set[str] = field(default_factory=set)
    memory_lines: list[str] = field(default_factory=list)
    memory_images: list[list[Any]] = field(default_factory=list)
    current_page_id: str | None = None
    current_evidence_ref: str | None = None
    current_image: Any | None = None
    current_query: str = ""
    current_observation_kind: str = ""
    root_sufficient: bool = False
    root_answer: str = ""
    pending_root_update: bool = False
    graph_facts: list[str] = field(default_factory=list)
    graph_rejections: list[str] = field(default_factory=list)
    graph_expansions: list[str] = field(default_factory=list)
    last_answer_reward: float | None = None
    last_retrieval_reward: float | None = None
    query_history: list[tuple[str, str]] = field(default_factory=list)
    # page_actions: page_label -> "accept"|"reject" from page commits, drives R_hit_all_gt.
    page_actions: dict[str, str] = field(default_factory=dict)
    last_grounding_reward: float | None = None
    # AGv2 crop context: <bbox> crops crop_source_image; crop-commit facts go to crop_target_node.
    # While a crop chain is live, graph.active is PINNED to crop_target_node; the deferred move
    # (accept -> parent / expand -> remaining child) resumes when the chain ends.
    crop_source_image: Any | None = None
    crop_displayed_size: tuple[int, int] | None = None
    crop_source_label: str = ""
    crop_target_node_id: str | None = None
    crop_resume_active_node_id: str | None = None
    step_observation: str = ""
    chat_turns: list[tuple[str, str, list[Any]]] = field(default_factory=list)
    current_user_content: str = ""
    current_user_images: list[Any] = field(default_factory=list)

    @property
    def anchor(self) -> tuple[str, ...]:
        return tuple(sorted(self.hit_pages))


class SlideVQAMultiProcessEnv:
    """Lightweight vectorized SlideVQA environment for verl-agent rollouts."""

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config: Any | None = None,
    ) -> None:
        self.seed = seed
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.env_config = env_config
        self.slidevqa_config = _cfg_get(env_config, "slidevqa", {})
        self.max_steps = int(_cfg_get(env_config, "max_steps", 5))
        self.top_k = int(_cfg_get(self.slidevqa_config, "top_k", 1))
        self.mode = str(_cfg_get(self.slidevqa_config, "mode", "ra")).lower()
        self.analysis_mode = str(_cfg_get(self.slidevqa_config, "analysis_mode", "page_label"))
        self.history_window = int(_cfg_get(self.slidevqa_config, "history_window", 2))
        self.observation_mode = str(_cfg_get(self.slidevqa_config, "observation_mode", "template")).lower()
        self.states: list[SlideVQAState] = []
        self.retriever = None
        self.vlm = None
        self.reward_fn = self._load_reward_fn()
        self.support_fn = self._load_support_fn()
        # AGv2: fact-box grounding retired (facts carry no bbox); grounding term is 0.
        self.grounding_w = float(os.environ.get("GROUNDING_W", "0.5"))
        self.retrieval_w = float(os.environ.get("RETRIEVAL_W", "0.5"))
        # Parallel env.step: per-state work (esp. the blocking DeepSeek answer-judge over HTTP) was
        # serial across the batch, so N trajectories answering in one rollout step cost N*judge
        # latency back-to-back (this is what tripped the 10min NCCL watchdog around step 35).
        # _step_one is dispatched across a thread pool; the shared CUDA retriever is serialized by a
        # lock so only the I/O-bound judge calls actually overlap. SLIDEVQA_STEP_WORKERS=1 -> serial.
        self._retriever_lock = threading.Lock()
        self._step_workers = max(1, int(os.environ.get("SLIDEVQA_STEP_WORKERS", "32")))

    def reset(self, kwargs: List[Dict[str, Any]] | None):
        if kwargs is None:
            kwargs = []
        elif isinstance(kwargs, np.ndarray):
            kwargs = kwargs.tolist()
        elif not isinstance(kwargs, list):
            kwargs = list(kwargs)
        if len(kwargs) > self.batch_size:
            raise ValueError(f"Got {len(kwargs)} kwarg dicts, but total_envs={self.batch_size}")

        self.states = [self._state_from_kwargs(item or {}) for item in kwargs]
        observations = [self._build_observation(state) for state in self.states]
        infos = [self._build_info(state, reward=0.0, action="reset") for state in self.states]
        return observations, infos

    def step(self, actions: List[str]):
        actions = list(actions)
        n = len(self.states)
        if self._step_workers > 1 and n > 1:
            # Overlap the per-trajectory work; the blocking answer-judge is the reason this matters.
            with ThreadPoolExecutor(max_workers=min(self._step_workers, n)) as pool:
                results = list(pool.map(self._step_one, self.states, actions))
        else:
            results = [self._step_one(state, action) for state, action in zip(self.states, actions)]
        observations = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        return observations, rewards, dones, infos

    def _step_one(self, state: SlideVQAState, action_text: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Advance one trajectory a single step. Extracted verbatim from the old serial step() loop
        so the batch can be dispatched concurrently across a thread pool (see __init__)."""
        if state.done:
            return (
                self._build_observation(state),
                0.0,
                True,
                self._build_info(state, reward=0.0, action="done"),
            )

        if self.mode == "active_graph":
            observation, reward, done, info = self._step_active_graph(state, action_text)
            return observation, float(reward), bool(done), info

        action, payload = parse_slidevqa_action(action_text)
        reward = 0.0
        retrieved_page_ids: list[str] = []
        hit_pages: list[str] = []

        if action == "answer":
            state.final_answer = payload
            state.done = True
            reward = self._score_terminal(state)
        elif action == "search":
            retrieved = self._search(payload, state.deck_name, state.seen_page_labels)
            retrieved_page_ids = [page_label for _, page_label in retrieved]
            for page_id in retrieved_page_ids:
                state.retrieved_page_history.append(page_id)
                if page_id in state.reference_pages and page_id not in state.hit_pages:
                    state.hit_pages.add(page_id)
                    hit_pages.append(page_id)
            state.memory_lines.append(self._memory_line(state, payload, retrieved))
            state.iter += 1
            if state.iter >= state.max_steps:
                state.done = True
                reward += self._score_terminal(state)
        else:
            state.memory_lines.append(
                f"[iter {state.iter + 1}] Invalid action format. Use exactly one <search>...</search> or <answer>...</answer> block."
            )
            reward = -1.0
            state.done = True
            state.iter += 1

        observation = self._build_observation(state)
        info = self._build_info(state, reward=reward, action=action)
        info["ra_is_answer"] = action == "answer"
        info["ra_action"] = action
        info["ra_retrieved_page_ids"] = retrieved_page_ids
        info["ra_hit_pages"] = hit_pages
        return observation, float(reward), bool(state.done), info

    def close(self):
        return None

    def _state_from_kwargs(self, item: dict[str, Any]) -> SlideVQAState:
        extra_info = item.get("extra_info") or {}
        deck_name = item.get("deck_name") or extra_info.get("deck_name")
        question = str(item.get("question") or extra_info.get("question") or "")
        answer = str(item.get("answer") or item.get("ground_truth") or extra_info.get("answer") or "")
        reference_pages = normalize_reference_pages(
            item.get("reference_pages")
            or item.get("page_labels")
            or item.get("evidence_pages")
            or extra_info.get("reference_pages")
            or extra_info.get("page_labels")
            or extra_info.get("evidence_pages"),
            deck_name=deck_name,
        )
        state = SlideVQAState(
            question=question,
            answer=answer,
            deck_name=str(deck_name) if deck_name else None,
            data_source=str(item.get("data_source") or extra_info.get("data_source") or "slidevqa"),
            reference_pages=reference_pages,
            max_steps=self.max_steps,
            sample_id=str(item.get("sample_id") or extra_info.get("sample_id") or extra_info.get("index") or ""),
            mode=self.mode,
            active_question=question,
        )
        if state.mode == "active_graph":
            self._ensure_active_graph_api()
            state.graph = ClueGraph.from_root_question(question)
            self._sync_graph_cache(state)
        return state

    def _ensure_active_graph_api(self) -> None:
        missing = [
            name
            for name, value in {
                "ClueGraph": ClueGraph,
                "ObservationImage": ObservationImage,
                "QueryRecord": QueryRecord,
                "VisualObservation": VisualObservation,
                "GraphDecisionResult": GraphDecisionResult,
                "apply_graph_decision": apply_graph_decision,
                "format_graph_state_for_prompt": format_graph_state_for_prompt,
                "format_observation_for_prompt": format_observation_for_prompt,
                "agv2_protocol": agv2,
            }.items()
            if value is None
        ]
        if missing:
            raise RuntimeError(
                "Active Graph RL could not import visual_rag_agent graph APIs: "
                + ", ".join(missing)
                + ". Ensure /scratch/punim0614/lifuzhang/visual_rag_agent is present."
            )

    def _active_graph(self, state: SlideVQAState) -> Any:
        self._ensure_active_graph_api()
        if state.graph is None:
            state.graph = ClueGraph.from_root_question(state.question)
        return state.graph

    def _sync_graph_cache(self, state: SlideVQAState) -> None:
        if state.graph is None:
            return
        graph = state.graph
        root = graph.root()
        active = graph.active()
        state.active_question = active.question
        state.root_sufficient = bool(is_sufficient(root))
        state.root_answer = str(root.answer or "")
        state.graph_facts = [(f.get("fact") if isinstance(f, dict) else str(f)) for f in root.known_facts]
        state.graph_expansions = [
            f"{graph.nodes[child_id].question} -> {graph.nodes[child_id].answer or 'unknown'}"
            for child_id in root.children
            if is_sufficient(graph.nodes[child_id])
        ]
        rejections: list[str] = []
        for node in graph.nodes.values():
            for record in node.query_history:
                if record.outcome in {"failure", "weak", "irrelevant", "unreadable"}:
                    rejections.append(record.summary or record.reason or record.query)
        state.graph_rejections = rejections

    def _build_observation(self, state: SlideVQAState) -> str:
        if state.mode == "active_graph":
            return self._build_active_graph_observation(state)
        memory = "\n".join(state.memory_lines) if state.memory_lines else "No pages searched yet."
        return SLIDEVQA_TEMPLATE.format(question=state.question, memory=memory)

    def _build_info(self, state: SlideVQAState, *, reward: float, action: str) -> dict[str, Any]:
        answer_reward = state.last_answer_reward if state.done else None
        retrieval_reward = state.last_retrieval_reward if state.done else None
        info = {
            "won": bool(answer_reward is not None and answer_reward >= 1.0),
            "task_score": float(reward),
            "data_source": state.data_source,
            "ra_step_kind": "decide",
            "ra_action": action,
            "ra_anchor": state.anchor,
            "ra_gt_pages": tuple(sorted(state.reference_pages)),
            "ra_missing_gt_pages": tuple(sorted(state.reference_pages - state.hit_pages)),
            "ra_hit_all_gt": self._hit_all_gt(state),
            "ra_final_answer": state.final_answer,
            "ra_answer_reward": answer_reward,
            "ra_retrieval_reward": retrieval_reward,
            "ra_grounding_reward": (state.last_grounding_reward if state.done else None),
            "reference_pages": tuple(sorted(state.reference_pages)),
            "retrieved_page_history": tuple(state.retrieved_page_history),
        }
        if state.mode == "active_graph":
            info["valid_actions"] = tuple(self._active_graph_valid_actions(state))
            info["observation_image"] = state.current_image
            if self.observation_mode == "multi_turn":
                info["observation_chat"] = self._build_active_graph_chat(state)
                info["observation_images"] = self._active_graph_chat_images(state)
                info["current_user_content"] = state.current_user_content or self._current_user_content(state)
                info["current_user_images"] = list(state.current_user_images or [])
                info["trajectory_user_messages"] = self._trajectory_user_messages(state)
                info["trajectory_user_images"] = self._trajectory_user_images(state)
            else:
                info["observation_images"] = self._query_history_images(state)
        return info

    def _format_observation_history(self, observation: Any | None) -> str:
        self._ensure_active_graph_api()
        if observation is None:
            return "Observation: None."
        text = format_observation_for_prompt(observation)
        if text.startswith("Current visual observation:"):
            return "Observation:" + text[len("Current visual observation:"):]
        return text

    def _observation_prompt_images(self, observation: Any | None) -> list[Any]:
        if observation is None:
            return []
        return list(observation.prompt_images())

    def _make_search_observation(
        self,
        state: SlideVQAState,
        *,
        query: str,
        retrieved: list[tuple[Any, str]],
    ) -> Any:
        self._ensure_active_graph_api()
        images: list[Any] = []
        skipped_duplicate_pages: list[str] = []
        for idx, (image, page_label) in enumerate(retrieved, start=1):
            label = str(page_label or f"retrieved_{idx}")
            if label in state.seen_page_labels:
                skipped_duplicate_pages.append(label)
                continue
            state.seen_page_labels.add(label)
            rgb = _image_to_rgb_array(image)
            image_id = make_image_id(label, idx)
            images.append(
                ObservationImage(
                    image_id=image_id,
                    page_label=label,
                    image=rgb,
                )
            )

        if not retrieved:
            message = "Search returned no visual candidates."
        elif skipped_duplicate_pages and not images:
            message = f"Search returned only duplicate pages: {skipped_duplicate_pages}."
        elif skipped_duplicate_pages:
            message = f"Skipped duplicate pages: {skipped_duplicate_pages}."
        else:
            message = ""

        return VisualObservation(query=query, images=images, kind="search", message=message)

    def _set_current_from_observation(self, state: SlideVQAState, observation: Any | None) -> None:
        state.observation = observation
        state.current_image = None
        state.current_page_id = None
        state.current_evidence_ref = None
        state.current_observation_kind = ""
        if observation is None:
            state.current_query = ""
            state.current_user_images = []
            return
        state.current_query = str(observation.query or "")
        state.current_observation_kind = str(observation.kind or "")
        if observation.images:
            image = observation.images[0]
            state.current_image = image.prompt_image()
            state.current_page_id = str(image.page_label or "")
            state.current_evidence_ref = str(image.evidence_ref() or image.image_id)
        state.current_user_images = self._observation_prompt_images(observation)

    def _step_active_graph(self, state: SlideVQAState, action_text: str) -> tuple[str, float, bool, dict[str, Any]]:
        """AGv2 merged-action step: <think> + [<update_graph> iff obs pending] + one action.
        Structural violations terminate with episode -1 (format_error lane); a malformed bbox
        payload is the soft lane (step -1, continue)."""
        observation_pending = state.observation is not None
        reward = 0.0
        retrieved_page_ids: list[str] = []
        hit_pages: list[str] = []
        info_extras: dict[str, Any] = {}
        state.step_observation = ""
        intended_group = "update_graph" if observation_pending else "search"

        def _format_error(detail: str) -> None:
            nonlocal reward
            reward = -1.0
            state.done = True
            state.step_observation = f"Invalid response: {detail}"
            self._append_memory_turn(state, f"[step {state.iter + 1}] INVALID: {detail}")
            info_extras.update({
                "ra_step_kind": "format_error", "format_error": True,
                "ra_step_reward": 0.0, "ra_step_group": intended_group,
                "ra_step_page": state.current_page_id or "",
                "ra_immediate_reward": 0.0,
            })

        turn = None
        soft_box_error: str | None = None
        try:
            turn = agv2.parse_turn(action_text, observation_pending=observation_pending)
        except agv2.BoxFormatError as exc:
            soft_box_error = str(exc)
        except agv2.ProtocolError as exc:
            _format_error(str(exc))

        action_type = "invalid"
        if soft_box_error is not None:
            # bbox payload malformed: step -1, no terminate; the (unparsed) update block is lost,
            # so the observation stays pending and the model must recommit next turn.
            action_type = "bbox"
            state.step_observation = (
                f"Invalid bbox: {soft_box_error}. Coordinates must be [x1,y1,x2,y2] "
                "pixels on the displayed image. The pending observation is still uncommitted."
            )
            self._append_memory_turn(state, f"[step {state.iter + 1}] {state.step_observation}")
            info_extras.update({
                "ra_step_kind": "box_error",
                "ra_step_reward": -1.0, "ra_step_group": "bbox",
                "ra_step_page": state.crop_source_label or "",
                "ra_immediate_reward": -1.0,
            })
            state.iter += 1
        elif turn is not None:
            action_type = turn.action if turn.action is not None else "commit"
            commit_ok = True
            if turn.update_payload is not None:
                commit_ok = self._commit_update(state, turn, info_extras, _format_error)
            if commit_ok and not state.done:
                if turn.action is None:
                    # Commit-only turn: action deferred so the next turn sees the updated
                    # graph rendered (multi-hop final-answer pattern). Consumes a step.
                    state.iter += 1
                elif turn.action == "answer":
                    state.final_answer = turn.action_payload.strip()
                    agv2.finalize_root(self._active_graph(state), state.final_answer)
                    self._sync_graph_cache(state)
                    state.done = True
                    reward = self._score_terminal(state)
                    info_extras.setdefault("ra_step_group", None)
                    info_extras["ra_immediate_reward"] = float(reward)
                elif turn.action == "search":
                    reward = self._do_search_action(
                        state, turn.action_payload.strip(), info_extras,
                        retrieved_page_ids, hit_pages,
                    )
                elif turn.action == "bbox":
                    self._do_bbox_action(state, turn.box, info_extras, _format_error)

        if not state.done and state.iter >= state.max_steps:
            state.done = True
            reward += self._score_terminal(state)

        self._record_chat_turn(state, action_text)
        info = self._build_info(state, reward=reward, action=action_type)
        info["ra_is_answer"] = action_type == "answer"
        info["ra_action"] = action_type
        info["ra_retrieved_page_ids"] = retrieved_page_ids
        info["ra_hit_pages"] = hit_pages
        info.update(info_extras)
        return self._build_observation(state), reward, state.done, info

    def _commit_update(
        self,
        state: SlideVQAState,
        turn: Any,
        info_extras: dict[str, Any],
        _format_error,
    ) -> bool:
        """Apply the <update_graph> block. Returns False when a format error terminated the
        episode. Page commits earn the type-match step reward; crop commits are neutral."""
        obs = state.observation
        obs_kind = str(getattr(obs, "kind", "") or "")
        page_id = state.current_page_id or ""
        try:
            decision = GraphDecisionResult.model_validate(turn.update_payload)
        except Exception as exc:
            _format_error(f"invalid update_graph payload: {exc}")
            state.iter += 1
            return False

        graph = self._active_graph(state)
        update_type = str(decision.type).lower()
        if obs_kind == "crop":
            try:
                agv2.commit_crop_decision(graph, state.crop_target_node_id, decision)
            except Exception as exc:
                _format_error(f"crop commit failed: {exc}")
                state.iter += 1
                return False
            # chain ends unless this turn zooms again: execute the deferred active move.
            if turn.action != "bbox":
                if state.crop_resume_active_node_id and state.crop_resume_active_node_id in graph.nodes:
                    graph.set_active(state.crop_resume_active_node_id)
                self._clear_crop_ctx(state)
            self._sync_graph_cache(state)
            state.step_observation = f"Crop evidence committed ({update_type})."
            self._append_memory_turn(
                state, f"[step {state.iter + 1}] crop commit {update_type} -> {state.crop_target_node_id}."
            )
            info_extras.update({
                "ra_step_kind": "crop_commit",
                "ra_step_reward": 0.0,
                "ra_step_group": "update_graph",
                "ra_step_page": state.crop_source_label or page_id,
                "ra_immediate_reward": 0.0,
            })
        else:
            expected_action = expected_update_graph_action(state, page_id)
            type_match = (
                (update_type == "reject")
                if expected_action == "reject"
                else (update_type in {"accept", "expand"})
            )
            defer = turn.action == "bbox"
            try:
                dtype, facts_node, resume_id = agv2.commit_page_decision(
                    graph, decision,
                    obs.summary() if obs is not None else "",
                    defer_active_shift=defer,
                )
            except Exception as exc:
                _format_error(f"update_graph apply failed: {exc}")
                state.iter += 1
                return False
            self._sync_graph_cache(state)
            # crop context armed ONLY when this turn zooms (accept/expand + bbox): active is
            # pinned on facts_node, the deferred move is stored for the chain end.
            if defer and dtype in {"accept", "expand"} and obs is not None and obs.images:
                img = obs.images[0]
                state.crop_source_image = img.image
                state.crop_displayed_size = _image_wh(img.prompt_image())
                state.crop_source_label = str(img.page_label or "")
                state.crop_target_node_id = facts_node
                state.crop_resume_active_node_id = resume_id
            else:
                self._clear_crop_ctx(state)
            if page_id:
                state.page_actions[page_id] = "reject" if dtype == "reject" else "accept"
            step_reward = 1.0 if type_match else 0.0
            serialized = decision.model_dump(mode="json", exclude_none=True)
            state.step_observation = (
                f"Graph update:\n{json.dumps(serialized, ensure_ascii=False, separators=(',', ':'))}"
            )
            self._append_memory_turn(state, f"[step {state.iter + 1}] {state.step_observation}")
            info_extras.update({
                "ra_step_kind": "analyse",
                "ra_analyse_reward": float(step_reward),
                "ra_step_reward": float(step_reward),
                "ra_step_group": "update_graph",
                "ra_step_page": page_id,
                "ra_step_type_match": bool(type_match),
                "ra_immediate_reward": float(step_reward),
                "ra_analyse_page_id": page_id,
                "ra_analyse_judge": update_graph_judge(update_type),
                "ra_analyse_expected_action": expected_action,
            })
        self._set_current_from_observation(state, None)
        return True

    def _do_search_action(
        self,
        state: SlideVQAState,
        query: str,
        info_extras: dict[str, Any],
        retrieved_page_ids: list[str],
        hit_pages: list[str],
    ) -> float:
        """Execute <search>: cursor retrieval (show first unseen), top1-new-GT step reward."""
        _full_retrieved = self._search(query, state.deck_name, None)
        _rank1_label = str(_full_retrieved[0][1]) if _full_retrieved else None
        _already_hit_before = set(state.hit_pages)
        _shown_pair = next(
            ((img, lbl) for (img, lbl) in _full_retrieved if str(lbl) not in state.seen_page_labels),
            None,
        )
        retrieved = [_shown_pair] if _shown_pair is not None else []
        for _, page_label in retrieved:
            retrieved_page_ids.append(page_label)
            state.retrieved_page_history.append(page_label)
            if page_label in state.reference_pages and page_label not in state.hit_pages:
                state.hit_pages.add(page_label)
                hit_pages.append(page_label)

        graph = self._active_graph(state)
        active_node = graph.active()
        active_node.query_history.append(QueryRecord(query=query, outcome="pending"))
        active_node.num_attempts += 1

        observation_obj = self._make_search_observation(state, query=query, retrieved=retrieved)
        self._set_current_from_observation(state, observation_obj)
        state.step_observation = self._format_observation_history(observation_obj)
        if observation_obj.images:
            state.query_history.append((query, observation_obj.images[0].page_label))
            self._append_memory_turn(
                state,
                f"[step {state.iter + 1}] SEARCH query={query!r}; current_observation={observation_obj.images[0].page_label}.\n{state.step_observation}",
                images=self._observation_prompt_images(observation_obj),
            )
        else:
            state.query_history.append((query, "no results"))
            self._append_memory_turn(
                state,
                f"[step {state.iter + 1}] SEARCH query={query!r}; no usable pages.\n{state.step_observation}",
            )
        # SEARCH step reward (TOP1): +1 iff this query's TRUE rank-1 (unfiltered) is a NEW GT
        # page — re-issuing the same query to advance the cursor earns nothing.
        _top1_new_gt = (
            _rank1_label is not None
            and _rank1_label in state.reference_pages
            and _rank1_label not in _already_hit_before
        )
        step_reward = 1.0 if _top1_new_gt else 0.0
        info_extras.update({
            "ra_search_reward": float(step_reward),
            "ra_step_reward": float(step_reward),
            "ra_step_group": "search",
            "ra_step_page": "",
            "ra_immediate_reward": float(step_reward),
        })
        state.iter += 1
        return 0.0

    def _do_bbox_action(
        self,
        state: SlideVQAState,
        box: list[float],
        info_extras: dict[str, Any],
        _format_error,
    ) -> None:
        """Execute <bbox>: crop the last accept/expand-committed page. No crop target =
        structural violation; a bad rectangle at execution time = soft step -1."""
        if state.crop_source_image is None or state.crop_target_node_id is None:
            _format_error("bbox requires a page committed with accept/expand first")
            state.iter += 1
            return
        try:
            source = state.crop_source_image
            pil = source if isinstance(source, Image.Image) else Image.fromarray(np.asarray(source))
            displayed = state.crop_displayed_size or pil.size
            crop_image = agv2.crop_displayed_box(pil, displayed, box)
        except agv2.BoxFormatError as exc:
            state.step_observation = f"Invalid bbox: {exc}."
            self._append_memory_turn(state, f"[step {state.iter + 1}] {state.step_observation}")
            info_extras.update({
                "ra_step_kind": "box_error",
                "ra_step_reward": -1.0, "ra_step_group": "bbox",
                "ra_step_page": state.crop_source_label or "",
                "ra_immediate_reward": -1.0,
            })
            state.iter += 1
            return
        crop_id = f"{make_image_id(state.crop_source_label, state.iter)}_crop_{state.iter}"
        observation_obj = VisualObservation(
            query="",
            kind="crop",
            images=[
                ObservationImage(
                    image_id=crop_id,
                    page_label=state.crop_source_label,
                    image=_image_to_rgb_array(crop_image),
                    crop_box=[float(v) for v in box],
                )
            ],
        )
        self._set_current_from_observation(state, observation_obj)
        state.step_observation = self._format_observation_history(observation_obj)
        self._append_memory_turn(
            state,
            f"[step {state.iter + 1}] BBOX {list(box)} on {state.crop_source_label}.\n{state.step_observation}",
            images=self._observation_prompt_images(observation_obj),
        )
        info_extras.update({
            "ra_step_kind": "bbox",
            "ra_step_reward": 0.0, "ra_step_group": "bbox",
            "ra_step_page": state.crop_source_label or "",
            "ra_immediate_reward": 0.0,
        })
        state.iter += 1

    def _build_active_graph_observation(self, state: SlideVQAState) -> str:
        if self.observation_mode == "multi_turn":
            state.current_user_content = self._current_user_content(state)
            state.current_user_images = []
            return state.current_user_content
        return SLIDEVQA_ACTIVE_GRAPH_TEMPLATE.format(
            question=state.question,
            active_question=state.active_question or state.question,
            graph_state=self._format_graph_state(state),
            query_history=self._format_query_history(state),
            valid_actions=agv2.pending_hint(state.observation is not None),
        )

    def _clear_crop_ctx(self, state: SlideVQAState) -> None:
        state.crop_source_image = None
        state.crop_displayed_size = None
        state.crop_source_label = ""
        state.crop_target_node_id = None
        state.crop_resume_active_node_id = None

    def _current_user_content(self, state: SlideVQAState) -> str:
        # AGv2: graph state + active question + pending-observation hint (mirrors eval's
        # build_turn_messages graph block exactly). No per-turn action mask.
        _content = f"Graph state:\n{self._format_graph_state_eval(state)}"
        _aq = state.active_question or state.question
        if _aq:
            _content = _content + chr(10) + "Question to answer right now (active node): " + str(_aq).strip()
        _pending_is_crop = state.observation is not None and str(getattr(state.observation, "kind", "")) == "crop"
        _content = _content + chr(10) + agv2.pending_hint(
            state.observation is not None,
            crop_page_label=state.crop_source_label if _pending_is_crop else "",
            crop_target_node=(state.crop_target_node_id or "") if _pending_is_crop else "",
        )
        return _content

    def _build_active_graph_chat(self, state: SlideVQAState) -> list[dict[str, str]]:
        if self.observation_mode != "multi_turn":
            return [{"role": "user", "content": self._build_active_graph_observation(state)}]
        current_graph = state.current_user_content or self._current_user_content(state)
        _sys = ACTIVE_GRAPH_POLICY_SYSTEM
        if state.question:
            _q = str(state.question).strip()
            _sys = ACTIVE_GRAPH_POLICY_SYSTEM + chr(10) + chr(10) + "=== CURRENT TASK - ORIGINAL QUESTION ===" + chr(10) + _q + chr(10) + "Keep this exact question as the goal of every action. Do NOT choose ANSWER until the graph holds supporting evidence for EVERY part of it; for multi-part or comparison questions, gather evidence for each part before answering."
        messages: list[dict[str, str]] = [{"role": "system", "content": _sys}]
        # Most recent round of dialogue only (action + its observation, image inline). Older rounds
        # live in the persistent graph state below.
        for assistant_text, user_text, _images in state.chat_turns[-1:]:
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "user", "content": user_text})
        messages.append({"role": "user", "content": current_graph})
        return messages

    def _active_graph_chat_images(self, state: SlideVQAState) -> list[Any]:
        if self.observation_mode != "multi_turn":
            return self._query_history_images(state)
        if state.chat_turns:
            return list(state.chat_turns[-1][2] or [])
        return []

    def _trajectory_user_messages(self, state: SlideVQAState) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if state.chat_turns:
            messages.append({"role": "user", "content": str(state.chat_turns[-1][1]).strip()})
        messages.append({"role": "user", "content": state.current_user_content or self._current_user_content(state)})
        return messages

    def _trajectory_user_images(self, state: SlideVQAState) -> list[Any]:
        if state.chat_turns:
            return list(state.chat_turns[-1][2] or [])
        return []

    def _record_chat_turn(self, state: SlideVQAState, assistant_text: str) -> None:
        if self.observation_mode != "multi_turn":
            return
        user_text = str(state.step_observation or "Observation: None.").strip()
        user_images = list(state.current_user_images or [])
        if not user_images and state.current_image is not None and "<image>" in user_text:
            user_images = [state.current_image]
        state.chat_turns.append((str(assistant_text or "").strip(), user_text, list(user_images)))
        state.current_user_content = self._current_user_content(state)
        state.current_user_images = []

    def _active_graph_valid_actions(self, state: SlideVQAState) -> list[str]:
        # AGv2: no per-turn action mask; kept only for the info dict (logging back-compat).
        if state.done:
            return []
        if state.observation is not None:
            return ["UPDATE_GRAPH+ACTION"]
        return ["ACTION"]

    def _format_graph_state(self, state: SlideVQAState) -> str:
        return self._format_graph_state_eval(state)

    def _format_graph_state_eval(self, state: SlideVQAState) -> str:
        graph = self._active_graph(state)
        # graphv2 everywhere now (train == eval). Single renderer, no local/graphv2 split.
        return format_graph_state_for_prompt(graph)

    def _append_memory_turn(self, state: SlideVQAState, text: str, images: list[Any] | None = None) -> None:
        state.memory_lines.append(str(text).strip())
        state.memory_images.append(list(images or []))

    def _history_slice(self, state: SlideVQAState) -> tuple[list[str], list[list[Any]]]:
        if self.history_window <= 0:
            return [], []
        lines = state.memory_lines[-self.history_window:]
        image_groups = state.memory_images[-len(lines):] if lines else []
        if len(image_groups) < len(lines):
            image_groups = [[] for _ in range(len(lines) - len(image_groups))] + image_groups
        return list(lines), [list(group or []) for group in image_groups]

    def _format_query_history(self, state: SlideVQAState) -> str:
        if self.history_window <= 0:
            return "No action-observation turns shown in the current sliding window."
        turns, _ = self._history_slice(state)
        if not turns:
            return "No previous action-observation turns yet."
        return "\n\n".join(turns)

    def _query_history_images(self, state: SlideVQAState) -> list[Any]:
        _, image_groups = self._history_slice(state)
        images: list[Any] = []
        for group in image_groups:
            images.extend(group)
        return images

    def _apply_graph_update(self, state: SlideVQAState, *, page_id: str, payload: dict[str, Any]) -> None:
        update_type = str(payload.get("type") or "").strip().lower()
        if update_type == "accept":
            answer = str(payload.get("answer") or "").strip()
            for fact in _facts_text(payload.get("supporting_facts")):
                state.graph_facts.append(f"{page_id}: {fact}")
            if state.pending_root_update or (state.active_question or state.question) == state.question:
                state.root_answer = answer
                state.root_sufficient = bool(answer)
                state.pending_root_update = False
                state.active_question = state.question
                self._append_memory_turn(
                    state,
                    f"[step {state.iter + 1}] UPDATE_GRAPH accept root from {page_id}; answer={answer!r}.",
                )
                return

            child_question = state.active_question or "subquestion"
            state.graph_expansions.append(f"{child_question} -> {answer or 'unknown'} [{page_id}]")
            state.pending_root_update = True
            state.active_question = state.question
            self._append_memory_turn(
                state,
                f"[step {state.iter + 1}] UPDATE_GRAPH accept child page={page_id}; answer={answer!r}; root update pending.",
            )
            return

        if update_type == "expand":
            answered = str(payload.get("answered_subquestion") or "").strip()
            answer = str(payload.get("answer") or "").strip()
            remaining = str(payload.get("remaining_subquestion") or "").strip()
            state.pending_root_update = False
            if answered or answer:
                state.graph_expansions.append(f"{answered or 'subquestion'} -> {answer or 'unknown'} [{page_id}]")
            for fact in _facts_text(payload.get("supporting_facts")):
                state.graph_facts.append(f"{page_id}: {fact}")
            if remaining:
                state.active_question = remaining
            self._append_memory_turn(
                state,
                f"[step {state.iter + 1}] UPDATE_GRAPH expand page={page_id}; remaining={remaining!r}.",
            )
            return

        summary = str(payload.get("summary") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        state.pending_root_update = False
        state.graph_rejections.append(f"{page_id}: {summary or reason or 'not useful'}")
        self._append_memory_turn(state, f"[step {state.iter + 1}] UPDATE_GRAPH reject page={page_id}; reason={reason!r}.")

    def _memory_line(self, state: SlideVQAState, query: str, retrieved: list[tuple[Any, str]]) -> str:
        labels = [page_label for _, page_label in retrieved]
        if not labels:
            return f"[iter {state.iter + 1}] search={query!r}; retrieved no pages."

        pieces = [f"[iter {state.iter + 1}] search={query!r}; retrieved: {', '.join(labels)}."]
        if self.analysis_mode == "vlm":
            analyses = self._analyse_pages(state, query, retrieved)
            if analyses:
                pieces.append("Analysis: " + " ".join(analyses))
        return " ".join(pieces)

    def _search(self, query: str, deck_name: str | None, seen_page_labels: "set | None" = None) -> list[tuple[Any, str]]:
        k = max(int(self.top_k), int(os.environ.get("ACTIVE_GRAPH_RETRIEVE_K", str(self.top_k))))
        # Serialize the shared CUDA retriever; sibling threads still overlap on the (HTTP) judge.
        with self._retriever_lock:
            self._ensure_retriever()
            retrieved = self.retriever.search(query, top_k=k, deck_name=deck_name)
        if k <= int(self.top_k) or seen_page_labels is None:
            return retrieved
        # top-k first-unseen (2026-06-13): return only the first page not already seen, so the
        # model still sees <=1 page AND hit_pages/search_reward are computed over the SHOWN page.
        for pair in retrieved:
            if str(pair[1]) not in seen_page_labels:
                return [pair]
        return []

    def _ensure_retriever(self) -> None:
        if self.retriever is not None:
            return
        visual_rag_agent_path = _cfg_get(self.slidevqa_config, "visual_rag_agent_path", None)
        if visual_rag_agent_path:
            path = str(Path(visual_rag_agent_path).expanduser().resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            from src.retriever import Retriever
        except Exception as exc:
            raise RuntimeError(
                "SlideVQA env could not import visual_rag_agent src.retriever. "
                "Set env.slidevqa.visual_rag_agent_path to the visual_rag_agent checkout."
            ) from exc

        self.retriever = Retriever(
            model_path=str(_cfg_get(self.slidevqa_config, "retriever_path", "Qwen/Qwen3-VL-Embedding-8B")),
            index_path=str(_cfg_get(self.slidevqa_config, "index_path", "data/indexes/slidevqa")),
            device=str(_cfg_get(self.slidevqa_config, "device", "cuda")),
            dtype=str(_cfg_get(self.slidevqa_config, "dtype", "bfloat16")),
            attn_implementation=_cfg_get(self.slidevqa_config, "attn_implementation", "flash_attention_2"),
        )

    def _analyse_pages(self, state: SlideVQAState, query: str, retrieved: list[tuple[Any, str]]) -> list[str]:
        self._ensure_vlm()
        outputs: list[str] = []
        memory = "\n".join(state.memory_lines) if state.memory_lines else "No memory yet."
        for image, page_label in retrieved:
            result = self.vlm.analyse(image=image, original_query=state.question, memory_context=memory)
            payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result)
            summary = payload.get("summary") or payload.get("answer_text") or payload.get("partial_answer") or ""
            key_facts = payload.get("key_facts") or payload.get("cited_statements") or []
            outputs.append(f"{page_label}: {summary} key_facts={key_facts}")
        return outputs

    def _ensure_vlm(self) -> None:
        if self.vlm is not None:
            return
        visual_rag_agent_path = _cfg_get(self.slidevqa_config, "visual_rag_agent_path", None)
        if visual_rag_agent_path:
            path = str(Path(visual_rag_agent_path).expanduser().resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            from src.vlm import VLM
        except Exception as exc:
            raise RuntimeError(
                "SlideVQA env could not import visual_rag_agent src.vlm for analysis_mode=vlm."
            ) from exc
        analysis_model_path = _cfg_get(self.slidevqa_config, "analysis_model_path", None)
        if not analysis_model_path:
            analysis_model_path = _cfg_get(self.slidevqa_config, "model_path", "Qwen/Qwen3-VL-Instruct-4B")
        self.vlm = VLM(
            model_path=str(analysis_model_path),
            device=str(_cfg_get(self.slidevqa_config, "device", "cuda")),
            dtype=str(_cfg_get(self.slidevqa_config, "dtype", "bfloat16")),
            attn_implementation=_cfg_get(self.slidevqa_config, "attn_implementation", "flash_attention_2"),
        )

    def _load_reward_fn(self) -> Callable[..., float]:
        reward_fn_path = _cfg_get(self.slidevqa_config, "reward_fn_path", None)
        if reward_fn_path:
            path = Path(str(reward_fn_path)).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                spec = importlib.util.spec_from_file_location("slidevqa_reward_fn", path)
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "compute_score"):
                        return module.compute_score
        return default_score

    def _load_support_fn(self):
        # judge_answer_and_support lives in the same reward module (compute_score path);
        # returns {correct, support, support_gold}. None -> correctness-only fallback.
        reward_fn_path = _cfg_get(self.slidevqa_config, "reward_fn_path", None)
        if reward_fn_path:
            path = Path(str(reward_fn_path)).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                spec = importlib.util.spec_from_file_location("slidevqa_support_fn", path)
                if spec is not None and spec.loader is not None:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "judge_answer_and_support"):
                        return module.judge_answer_and_support
        return None

    def _score_answer(self, state: SlideVQAState) -> float:
        return float(
            self.reward_fn(
                data_source=state.data_source,
                solution_str=state.final_answer,
                ground_truth=state.answer,
                extra_info={
                    "sample_id": state.sample_id,
                    "question": state.question,
                    "deck_name": state.deck_name,
                    "reference_pages": tuple(sorted(state.reference_pages)),
                    "retrieved_page_history": tuple(state.retrieved_page_history),
                },
            )
        )

    def _hit_all_gt(self, state: SlideVQAState) -> bool:
        # True iff EVERY GT page has been update_graph-accepted (accept OR expand — envs.py marks
        # both as page_actions[g]=="accept"). Stricter than SEARCH-retrieval: rewards the agent
        # only when it committed every GT page to the graph (not just retrieved it).
        if not state.reference_pages:
            return False
        return all(state.page_actions.get(g) == "accept" for g in state.reference_pages)

    def _score_retrieval(self, state: SlideVQAState) -> float:
        # 1.0 if model accepted all GT pages by terminal; 0.0 otherwise. Weighted by RETRIEVAL_W in terminal.
        return 1.0 if self._hit_all_gt(state) else 0.0

    def _judge_answer_and_support(self, state: SlideVQAState) -> tuple[bool, bool, bool]:
        # ONE DeepSeek call -> (correct, support, support_gold): correct = answer matches gold;
        # support = collected facts entail the model PREDICTION; support_gold = facts entail the GOLD answer.
        # Fallbacks: no answer/gold -> (F,F,F); reward module without the combined judge -> correctness-only.
        prediction = str(state.final_answer or "").strip()
        gold = str(state.answer or "").strip()
        if not prediction or not gold:
            return False, False, False
        if self.support_fn is None:
            return (self._score_answer(state) > 0), False, False
        all_facts = []
        if state.graph is not None:
            for _node in state.graph.nodes.values():
                for _f in (getattr(_node, "known_facts", None) or []):
                    all_facts.append(_f.get("fact") if isinstance(_f, dict) else str(_f))
        try:
            judged = self.support_fn(
                question=state.question,
                prediction=prediction,
                gold_answer=gold,
                facts=all_facts,
                sample_id=state.sample_id,
            )
            correct = bool(judged.get("correct") is True and int(judged.get("score", 0)) == 1)
            support = bool(judged.get("support") is True)
            support_gold = bool(judged.get("support_gold") is True)
            return correct, support, support_gold
        except Exception as exc:  # noqa: BLE001
            print(f"[support_judge_error] {type(exc).__name__}: {exc}", flush=True)
            return False, False, False

    def _score_grounding(self, state: SlideVQAState) -> float:
        # AGv2: facts carry no bbox, so fact-box R_grounding is retired (returns 0; the
        # GROUNDING_W * 0 term keeps the episode formula shape). If a grounding signal is
        # wanted later, score the <bbox> action boxes against the teacher key-fact boxes here.
        return 0.0

    def _score_terminal(self, state: SlideVQAState) -> float:
        # episode reward (2026-06-29 v3 = 3-term: answer + W_g * R_grounding + W_r * R_hit_all_gt):
        #   no-answer (empty prediction) -> 0
        #   correct                      -> 1 + W_g * R_grounding + W_r * R_hit_all_gt
        #   wrong                        -> 0 + W_g * R_grounding + W_r * R_hit_all_gt
        #   default W_g=0.5 W_r=0.5: correct ∈ [1.0, 2.0]; wrong ∈ [0.0, 1.0]; worst-correct = best-wrong at 1.0 (correct >= wrong).
        #   format/structural -1+terminate is set by the caller, NOT here.
        # R_hit_all_gt = 1 if all teacher gold pages were accepted by the agent, else 0 (encourages finding all evidence, not just answering fast).
        prediction = str(state.final_answer or "").strip()
        # AGv2 multi-hop gate (D8): <answer> implicitly finalizes the root, so the old
        # "expanded but never accepted the root" check is replaced by "expanded but left an
        # open subquestion unresolved". Trajectories that never expanded are NOT punished
        # (under-decomposition is a quality issue, not a format violation).
        if os.environ.get("AG_FORMAT_GATE", "1") == "1" and len(state.reference_pages) > 1 and state.graph is not None:
            try:
                _has_expanded = bool(getattr(state.graph.root(), "children", None))
                _open_left = bool(agv2.open_nonroot_nodes(state.graph))
            except Exception:
                _has_expanded = False; _open_left = False
            if _has_expanded and _open_left:
                state.last_answer_reward = 0.0
                state.last_grounding_reward = 0.0
                state.last_retrieval_reward = 0.0
                return -1.0
        rgr = self._score_grounding(state)
        state.last_grounding_reward = float(rgr)
        retrieval_reward = self._score_retrieval(state)
        state.last_retrieval_reward = float(retrieval_reward)  # logging only
        if not prediction:
            state.last_answer_reward = 0.0
            return 0.0  # no-answer
        correct, support, support_gold = self._judge_answer_and_support(state)
        state.last_answer_reward = 1.0 if correct else 0.0
        base = 1.0 if correct else 0.0
        return float(base + self.grounding_w * rgr + self.retrieval_w * retrieval_reward)


def parse_slidevqa_action(action_text: str) -> tuple[str, str]:
    stripped = (action_text or "").lstrip()
    # Parse the model's generation as-is (no prefill, no reconstruction): strip a leading
    # <think>...</think> if present, otherwise just scan the whole text for the single action tag.
    think_match = re.match(
        r"<\s*think\s*>(.*?)<\s*/\s*think\s*>",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if think_match is not None and think_match.start() == 0 and think_match.group(1).strip():
        rest = stripped[think_match.end():].strip()
    else:
        rest = stripped
    matches: list[tuple[int, int, str, str]] = []
    for tag in ("search", "update_graph", "answer"):
        for match in re.finditer(rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>", rest, flags=re.IGNORECASE | re.DOTALL):
            matches.append((match.start(), match.end(), tag, match.group(1).strip()))
    if len(matches) != 1:
        return "invalid", ""
    start, end, tag, payload = matches[0]
    if rest[:start].strip() or rest[end:].strip():
        return "invalid", ""
    return tag, payload

def _array_to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(_image_to_rgb_array(image)).convert("RGB")


def _image_wh(image: Any) -> tuple[int, int] | None:
    """(width, height) for either a PIL image or an HxWxC numpy array."""
    if isinstance(image, Image.Image):
        return tuple(image.size)
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        return (int(shape[1]), int(shape[0]))
    return None


def update_graph_judge(update_type: str) -> str:
    if update_type == "accept":
        return "yes"
    if update_type == "expand":
        return "partial"
    if update_type == "reject":
        return "no"
    return "invalid"


def expected_update_graph_action(state: SlideVQAState, page_id: str) -> str:
    graph = state.graph
    if graph is not None:
        if state.observation is None:
            return "accept"
        if not state.observation.images:
            return "reject"
        if page_id not in state.reference_pages:
            return "reject"
        # No parent_id shortcut: with EXPAND-collapse a parent can re-expand, so "gt page +
        # pool not yet complete" -> expand at any node; only complete pool -> accept.
        if state.reference_pages and state.reference_pages.issubset(state.hit_pages):
            return "accept"
        return "expand"

    if state.pending_root_update:
        return "accept"
    if state.current_observation_kind and not page_id:
        return "reject"
    if page_id not in state.reference_pages:
        return "reject"
    if state.reference_pages and state.reference_pages.issubset(state.hit_pages):
        return "accept"
    return "expand"


def make_image_id(page_label: str, idx: int) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(page_label)).strip("_")
    safe = safe or f"image_{idx}"
    return safe[:96]


def _image_to_rgb_array(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray):
        array = image
    elif hasattr(image, "convert"):
        pil_image = image.convert("RGB")
        if pil_image.width <= 0 or pil_image.height <= 0:
            raise ValueError(f"Empty SlideVQA observation image: {(pil_image.width, pil_image.height)}")
        array = np.array(pil_image)
    else:
        array = np.array(image)

    if array.size == 0 or (array.ndim >= 2 and 0 in array.shape[:2]):
        raise ValueError(f"Empty SlideVQA observation image array: {array.shape}")
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = array.astype(np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape for SlideVQA observation: {array.shape}")
    return array


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    texts: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            texts.append(text)
    return texts


def normalize_reference_pages(value: Any, *, deck_name: str | None) -> set[str]:
    pages: set[str] = set()
    if value is None:
        return pages
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return pages
        if stripped.startswith("[") and stripped.endswith("]"):
            import json
            try:
                return normalize_reference_pages(json.loads(stripped), deck_name=deck_name)
            except Exception:
                pass
        return {stripped}
    for page in value:
        if isinstance(page, int):
            pages.add(format_page_label(deck_name, page))
            continue
        try:
            page_int = int(page)
        except (TypeError, ValueError):
            pages.add(str(page))
        else:
            pages.add(format_page_label(deck_name, page_int))
    return pages


def format_page_label(deck_name: str | None, page: int) -> str:
    label = f"page_{page:02d}"
    return f"{deck_name}/{label}" if deck_name else label


def default_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict[str, Any] | None = None) -> float:
    return 1.0 if str(solution_str).strip().lower() == str(ground_truth).strip().lower() else -1.0


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def build_slidevqa_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config: Any | None = None,
):
    return SlideVQAMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
    )


def _facts_text(value):
    """List[str] of fact labels from supporting_facts entries (SupportingFact/dict/str)."""
    out = []
    for f in (value or []):
        if isinstance(f, dict):
            t = str(f.get("fact") or f.get("label") or f.get("text") or "").strip()
        elif isinstance(f, str):
            t = f.strip()
        else:
            t = str(getattr(f, "fact", "") or "").strip()
        if t:
            out.append(t)
    return out
