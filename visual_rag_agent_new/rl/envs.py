from __future__ import annotations

"""verl-agent environment for the new protocol (drop into
verl-agent/agent_system/environments/env_package/slidevqa/envs.py).

Protocol, memory and per-turn context come from the visual_rag_agent_new checkout (src/protocol.py,
src/memory.py, src/prompts.py, src/agent.prepare_action), so RL rollouts and eval build identical
prompts. Point VISUAL_RAG_AGENT_PATH at that checkout; Ray rollout workers do not inherit the
launcher's PYTHONPATH, so the path is inserted here at import time.

Reward (DESIGN.md §8, §11):
  format error  -> -1, terminate
  budget timeout -> -1 if G is disabled or its final continuation is unusable
  soft error    -> step penalty AG_SOFT_ERROR_REWARD (default -1), the turn is not applied
  terminal      -> 1[correct] + COVERAGE_W*cov + FULL_COVERAGE_W*hit_all
                   + CMP_W*1[cov=1]*cmp + GROUNDING_W*R_grounding
                   - ACCESS_COST_W*(searches + revisits + crops)/max_steps
                   + POST_GOLD_SEARCH_W * 1[correct, final covers gold, all gold shown]
                     * POST_GOLD_SEARCH_DECAY ** searches_after_all_gold_shown
  cov / cmp are computed on E_final (the pages attached to the question tree, DESIGN.md §11.8), not on
  visited pages. COVERAGE_MODE=fraction (default: gold pages in E_final / gold pages) or all (1 only when every
  gold page is in E_final, else 0: the old v9 R_hit_all_gt, DESIGN.md §11.31).
  hit_all requires nonempty gold pages and every gold page in E_final. FULL_COVERAGE_W defaults to 0.
  Explicit env.slidevqa reward fields take precedence over their uppercase environment variables.
  Search decay excludes the search that finds the final missing page, crops and revisits.
  Automatic question search and every ordinary decision consume the hard max_steps budget.
Retrieval: one retriever per corpus (env.slidevqa.index_paths "train=<dir>,val=<dir>", sharing one ColQwen model);
a question searches the corpus named by its "corpus" kwarg (default "train"). A single env.slidevqa.index_path is the
"train" corpus.
  No positive step rewards (first version).
"""

import importlib.util
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

for _candidate in (
    os.environ.get("VISUAL_RAG_AGENT_PATH"),
    "/root/autodl-tmp/visual_rag_agent_new",
    "/scratch/punim0614/lifuzhang/visual_rag_agent_new",
):
    if _candidate and Path(_candidate).exists() and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from src.agent import FIRST_SEARCH, NO_NEW_PAGE_NOTE, prepare_action  # noqa: E402
from src.forced_answer import PREFIX, build_messages as build_forced_answer_messages
from src.memory import Memory, View, Visit, apply_edit  # noqa: E402
from src.prompts import build_turn_messages, first_search_response, turn_of  # noqa: E402
from src.protocol import get_frame  # noqa: E402
from src.protocol import EditError, ProtocolError, parse_turn, render_turn  # noqa: E402


@dataclass
class EpisodeState:
    question: str
    answer: str
    deck_name: str | None
    data_source: str
    reference_pages: set[str]
    keyfacts: list[dict[str, Any]]
    max_steps: int
    sample_id: str
    memory: Memory
    corpus: str = "train"
    view: View | None = None
    last_response: str | None = None
    note: str = ""
    turn: int = 0
    done: bool = False
    terminated_by: str = ""
    final_answer: str = ""
    final_pages: list[str] = field(default_factory=list)   # real labels of E_final
    budget_exhausted: bool = False
    forced_answer: bool = False
    counts: Counter = field(default_factory=Counter)
    retrieved_page_history: list[str] = field(default_factory=list)
    hit_pages: set[str] = field(default_factory=set)
    seen_pages: set[str] = field(default_factory=set)
    first_all_gold_seen_turn: int | None = None
    post_gold_searches: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)
    images: list[Image.Image] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def anchor(self) -> tuple[str, ...]:
        return tuple(sorted(self.hit_pages))


class SlideVQAMultiProcessEnv:
    """Vectorized environment for verl-agent rollouts (one EpisodeState per trajectory)."""

    def __init__(self, seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True, env_config: Any | None = None) -> None:
        self.seed = seed
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.env_config = env_config
        self.slidevqa_config = _cfg_get(env_config, "slidevqa", {})
        self.max_steps = int(_cfg_get(env_config, "max_steps", 10))
        self.forced_answer_prefill = bool(_cfg_get(self.slidevqa_config, "forced_answer_prefill", False))
        self.retrieve_k = int(os.environ.get("ACTIVE_GRAPH_RETRIEVE_K", _cfg_get(self.slidevqa_config, "top_k", 5)))
        # flat: one corpus over all decks (SCORE setting); deck: search only the question's deck
        self.retrieval_scope = str(_cfg_get(self.slidevqa_config, "retrieval_scope", "flat"))
        # the policy's native coordinate frame (DESIGN.md §11.3); images are shown at its size
        self.frame = get_frame(str(_cfg_get(self.slidevqa_config, "frame", "qwen2.5-vl")))
        # experiment switches, as in src.agent.Agent (DESIGN.md §11.5)
        self.first_search = str(_cfg_get(self.slidevqa_config, "first_search", "model"))
        if self.first_search not in FIRST_SEARCH:
            raise ValueError(f"first_search must be one of {FIRST_SEARCH}, got {self.first_search!r}")
        if self.max_steps <= int(self.first_search == "question"):
            raise ValueError("max_steps must leave at least one model turn after the automatic search")
        self.search_guidance = bool(_cfg_get(self.slidevqa_config, "search_guidance", False))
        if self.retrieval_scope not in ("flat", "deck"):
            raise ValueError(f"retrieval_scope must be flat or deck, got {self.retrieval_scope!r}")
        self.soft_error_reward = float(os.environ.get("AG_SOFT_ERROR_REWARD", "-1"))
        # Explicit experiment overrides must win over exports in an existing launcher.
        def reward_option(name: str, default: str) -> Any:
            return _cfg_get(self.slidevqa_config, name, os.environ.get(name.upper(), default))

        self.coverage_w = float(reward_option("coverage_w", "0.5"))
        self.coverage_mode = str(reward_option("coverage_mode", "fraction"))
        if self.coverage_mode not in ("fraction", "all"):
            raise ValueError(f"COVERAGE_MODE must be fraction or all, got {self.coverage_mode!r}")
        self.full_coverage_w = float(reward_option("full_coverage_w", "0.0"))
        self.cmp_w = float(reward_option("cmp_w", "0.2"))
        self.grounding_w = float(reward_option("grounding_w", "0.5"))
        self.access_cost_w = float(reward_option("access_cost_w", "0.1"))
        self.post_gold_search_w = float(reward_option("post_gold_search_w", "0.0"))
        self.post_gold_search_decay = float(reward_option("post_gold_search_decay", "0.5"))
        if not 0.0 < self.post_gold_search_decay <= 1.0:
            raise ValueError("post_gold_search_decay must be in (0, 1]")
        self.states: list[EpisodeState] = []
        self.retriever: Any | None = None          # the "train" corpus when only one index is configured
        self.retrievers: dict[str, Any] = {}         # corpus -> retriever
        self.reward_fn = self._load_reward_module("compute_score") or default_score
        self.support_fn = self._load_reward_module("judge_answer_and_support")
        # The shared CUDA retriever is serialized; the HTTP judge calls overlap across the pool.
        self._retriever_lock = threading.Lock()
        self._step_workers = max(1, int(os.environ.get("SLIDEVQA_STEP_WORKERS", "32")))

    # ------------------------------------------------------------------ gym-ish API

    def reset(self, kwargs: list[dict[str, Any]] | None):
        if kwargs is None:
            kwargs = []
        elif isinstance(kwargs, np.ndarray):
            kwargs = kwargs.tolist()
        elif not isinstance(kwargs, list):
            kwargs = list(kwargs)
        if len(kwargs) > self.batch_size:
            raise ValueError(f"Got {len(kwargs)} kwarg dicts, but total_envs={self.batch_size}")
        self.states = [self._state_from_kwargs(item or {}) for item in kwargs]
        for state in self.states:
            if self.first_search == "question":
                self._first_search(state)
            self._refresh_context(state)
        return [self._observation(s) for s in self.states], [self._build_info(s, reward=0.0, action="reset") for s in self.states]

    def step(self, actions: list[str]):
        actions = list(actions)
        n = len(self.states)
        if self._step_workers > 1 and n > 1:
            with ThreadPoolExecutor(max_workers=min(self._step_workers, n)) as pool:
                results = list(pool.map(self._step_one, self.states, actions))
        else:
            results = [self._step_one(s, a) for s, a in zip(self.states, actions)]
        return ([r[0] for r in results], [r[1] for r in results], [r[2] for r in results], [r[3] for r in results])

    def close(self):
        return None

    # ------------------------------------------------------------------ state

    def _state_from_kwargs(self, item: dict[str, Any]) -> EpisodeState:
        extra = item.get("extra_info") or {}
        deck_name = item.get("deck_name") or extra.get("deck_name")
        question = str(item.get("question") or extra.get("question") or "")
        answer = str(item.get("answer") or item.get("ground_truth") or extra.get("answer") or "")
        reference_pages = normalize_reference_pages(
            item.get("reference_pages") or item.get("page_labels") or item.get("evidence_pages")
            or extra.get("reference_pages") or extra.get("page_labels") or extra.get("evidence_pages"),
            deck_name=deck_name,
        )
        return EpisodeState(
            question=question,
            answer=answer,
            deck_name=str(deck_name) if deck_name else None,
            data_source=str(item.get("data_source") or extra.get("data_source") or "slidevqa"),
            reference_pages=reference_pages,
            keyfacts=normalize_keyfacts(item.get("keyfacts") or extra.get("keyfacts"), deck_name=deck_name),
            max_steps=self.max_steps,
            sample_id=str(item.get("sample_id") or extra.get("sample_id") or extra.get("index") or ""),
            memory=Memory(question),
            corpus=str(item.get("corpus") or extra.get("corpus") or "train"),
        )

    def _first_search(self, state: EpisodeState) -> None:
        """Turn 0 is a fixed search with the question itself; it counts as an access and a turn."""
        self._do_search(state, state.question, {})
        state.counts["search"] += 1
        state.last_response = first_search_response(state.question)
        self._advance(state)

    def _refresh_context(self, state: EpisodeState) -> None:
        if state.forced_answer:
            state.messages = build_forced_answer_messages(state.memory.as_serializable())
            state.images = []
            return
        state.messages, state.images = build_turn_messages(
            state.memory, state.view, state.last_response, frame=self.frame, search_guidance=self.search_guidance,
            note=state.note, force=False,
            turn_of=turn_of(state.turn, state.max_steps, self.first_search))
        # Count pages actually placed in a model observation, not all retrieved candidates.
        if state.view is not None and state.images:
            state.seen_pages.add(state.memory.pages[state.view.page].label)
        if (state.first_all_gold_seen_turn is None and state.reference_pages
                and state.reference_pages <= state.seen_pages):
            state.first_all_gold_seen_turn = state.turn
        state.note = ""

    def _observation(self, state: EpisodeState) -> str:
        return next((str(msg["content"]) for msg in reversed(state.messages) if msg["role"] == "user"), "")

    # ------------------------------------------------------------------ stepping

    def _step_one(self, state: EpisodeState, action_text: str) -> tuple[str, float, bool, dict[str, Any]]:
        if state.done:
            return self._observation(state), 0.0, True, self._build_info(state, reward=0.0, action="done")
        extras: dict[str, Any] = dict(STEP_DEFAULTS)
        reward = self._step(state, action_text, extras)
        if not state.done:
            self._refresh_context(state)
        info = self._build_info(state, reward=reward, action=str(extras["ra_action"]))
        info.update(extras)
        return self._observation(state), float(reward), bool(state.done), info

    def _step(self, state: EpisodeState, text: str, extras: dict[str, Any]) -> float:
        if state.forced_answer:
            answer = text.split("</answer>", 1)[0].strip()
            if not answer or any(tag in answer for tag in ("<act", "<page", "<tree", "<think", PREFIX)):
                return self._timeout(state, extras)
            extras.update(ra_action="answer", ra_step_kind="answer", ra_step_group="answer")
            reward = self._finish(state, answer)
            extras["ra_immediate_reward"] = reward
            return reward
        view = state.view
        try:
            turn = parse_turn(text, view="none" if view is None else view.kind,
                              current_page=view.page if view else None, force_answer=False)
        except ProtocolError as exc:
            extras.update({"ra_step_kind": "format_error", "format_error": True, "format_error_detail": str(exc)})
            state.done = True
            state.terminated_by = "format_error"
            return -1.0
        act = turn.act
        target = render_turn(turn.think, act)     # the canonical response, the next turn's context (as src/agent.py)
        extras["ra_action"] = act.action.type
        try:
            new_m, events = apply_edit(state.memory, act.page, act.tree, view, turn=state.turn, frame=self.frame)
            zoom = prepare_action(new_m, view, act, self.frame, turn=state.turn)
        except EditError as exc:
            extras.update({"ra_step_kind": "edit_error", "edit_error_detail": str(exc), "ra_step_group": act.action.type,
                           "ra_step_reward": self.soft_error_reward, "ra_immediate_reward": self.soft_error_reward})
            state.note = f"your previous response was NOT applied ({exc}). Fix it and respond again."
            state.last_response = target
            self._advance(state)
            return self._on_budget_exhausted(state, extras) if state.turn >= state.max_steps else 0.0
        state.memory = new_m
        state.last_response = target
        action = act.action
        extras.update({"ra_step_kind": action.type, "ra_step_group": action.type,
                       "ra_resolved": tuple(events["resolved"]) + tuple(events["expanded"]),
                       "ra_step_page": view.page if view else ""})
        if action.type == "answer":
            reward = self._finish(state, action.text)
            extras["ra_immediate_reward"] = reward
            return reward
        state.counts[action.type] += 1
        if action.type == "search" and state.first_all_gold_seen_turn is not None:
            state.post_gold_searches += 1
        # The last ordinary turn can edit memory but cannot execute another tool.
        if state.turn + 1 >= state.max_steps:
            self._advance(state)
            return self._on_budget_exhausted(state, extras)
        if action.type == "crop":
            view.zoom = zoom
        elif action.type == "revisit":
            state.memory.pages[action.page_id].visits.append(Visit(state.turn, state.memory.active_id, "revisit"))
            state.view = View(page=action.page_id, via="revisit")
        else:
            self._do_search(state, action.query, extras)
        self._advance(state)
        return 0.0

    def _advance(self, state: EpisodeState) -> None:
        state.turn += 1

    def _on_budget_exhausted(self, state: EpisodeState, extras: dict[str, Any]) -> float:
        if self.forced_answer_prefill:
            state.budget_exhausted = state.forced_answer = True
            return 0.0
        return self._timeout(state, extras)

    def _timeout(self, state: EpisodeState, extras: dict[str, Any]) -> float:
        state.done = state.budget_exhausted = True
        state.terminated_by = "budget_timeout"
        state.final_pages = [state.memory.pages[p].label for p in state.memory.evidence_pages()]
        extras["ra_immediate_reward"] = -1.0
        return -1.0

    def _finish(self, state: EpisodeState, answer: str) -> float:
        state.done = True
        state.terminated_by = "answer"
        state.final_answer = answer
        if answer:
            state.memory.finish(answer)
        state.final_pages = [state.memory.pages[p].label for p in state.memory.evidence_pages()]
        return self._score_terminal(state)

    def _do_search(self, state: EpisodeState, query: str, extras: dict[str, Any]) -> None:
        with self._retriever_lock:
            retriever = self._retriever_for(state.corpus)
            deck = state.deck_name if self.retrieval_scope == "deck" else None
            retrieved = retriever.search(query, top_k=self.retrieve_k, deck_name=deck)
        m = state.memory
        shown = next(((img, str(lbl)) for img, lbl in retrieved if str(lbl) not in m.aliases), None)
        if shown is None:
            state.view = None
            state.note = NO_NEW_PAGE_NOTE
            extras["ra_retrieved_page_ids"] = ()
            return
        image, label = shown
        page = m.add_page(label, _to_pil(image), query)
        page.visits.append(Visit(state.turn, m.active_id, "search", query))
        state.view = View(page=page.id, via="search", query=query)
        state.retrieved_page_history.append(label)
        if label in state.reference_pages:
            state.hit_pages.add(label)
        extras["ra_retrieved_page_ids"] = (label,)

    def _retriever_for(self, corpus: str) -> Any:
        """The retriever of a corpus; KeyError for a corpus that has no index (never a silent fallback)."""
        if not self.retrievers and self.retriever is None:
            self._ensure_retriever()
        if self.retrievers:
            if corpus not in self.retrievers:
                raise KeyError(f"no index for corpus {corpus!r}; configured: {sorted(self.retrievers)}")
            return self.retrievers[corpus]
        if corpus != "train":
            raise KeyError(f"no index for corpus {corpus!r}; only one index_path is configured (the train corpus)")
        return self.retriever

    def _ensure_retriever(self) -> None:
        """Load the retrievers once per process: every env with the same retriever config (verl-agent's train and val
        envs) shares them (_RETRIEVERS), so the encoder and page banks are never loaded twice on the GPU."""
        if self.retriever is not None or self.retrievers:
            return
        cfg = self.slidevqa_config
        key = json.dumps({k: str(_cfg_get(cfg, k, None)) for k in
                          ("retriever_path", "index_path", "index_paths", "device", "dtype", "attn_implementation", "bank_device")},
                         sort_keys=True)
        with _RETRIEVERS_LOCK:
            if key not in _RETRIEVERS:
                _RETRIEVERS[key] = _build_retrievers(cfg)
            self.retrievers, self.retriever = _RETRIEVERS[key]

    # ------------------------------------------------------------------ rewards

    def _load_reward_module(self, attr: str) -> Callable[..., Any] | None:
        reward_fn_path = _cfg_get(self.slidevqa_config, "reward_fn_path", None)
        if not reward_fn_path:
            return None
        path = Path(str(reward_fn_path)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return None
        spec = importlib.util.spec_from_file_location(f"slidevqa_reward_{attr}", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, attr, None)

    def _final_facts(self, state: EpisodeState, src: str | None = None) -> list[tuple[str, Any]]:
        """(real page label, fact) for the facts of the E_final pages."""
        m = state.memory
        out = []
        for alias in m.evidence_pages():
            page = m.pages[alias]
            out.extend((page.label, f) for f in page.facts if src is None or f.src == src)
        return out

    def _judge_correct(self, state: EpisodeState) -> bool:
        prediction, gold = state.final_answer.strip(), state.answer.strip()
        if not prediction or not gold:
            return False
        if self.support_fn is None:
            score = self.reward_fn(
                data_source=state.data_source, solution_str=prediction, ground_truth=gold,
                extra_info={"sample_id": state.sample_id, "question": state.question, "deck_name": state.deck_name,
                            "reference_pages": tuple(sorted(state.reference_pages)),
                            "retrieved_page_history": tuple(state.retrieved_page_history)},
            )
            return float(score) > 0
        try:
            judged = self.support_fn(question=state.question, prediction=prediction, gold_answer=gold,
                                     facts=[f.text for _, f in self._final_facts(state)], sample_id=state.sample_id)
            return bool(judged.get("correct") is True and int(judged.get("score", 0)) == 1)
        except Exception as exc:  # noqa: BLE001
            print(f"[support_judge_error] {type(exc).__name__}: {exc}", flush=True)
            return False

    def _score_grounding(self, state: EpisodeState) -> float:
        """Mean over gold key-fact boxes of the best IoU with a page-frame fact (src == "page") on
        the same page among the E_final pages. Zoom-read boxes do not count. No keyfacts -> 0."""
        if not state.keyfacts:
            return 0.0
        preds = self._final_facts(state, src="page")
        scores = []
        for kf in state.keyfacts:
            scores.append(max((iou(kf["bbox_2d"], f.bbox) for label, f in preds if label == kf["page"]), default=0.0))
        return float(sum(scores) / len(scores))

    def _score_terminal(self, state: EpisodeState) -> float:
        final = set(state.final_pages)
        gold = state.reference_pages
        hit = len(gold & final)
        cov = hit / len(gold) if gold else 0.0
        hit_all = float(bool(gold) and hit == len(gold))
        if self.coverage_mode == "all":
            cov = hit_all
        cmp = hit / len(final) if final else 0.0
        grounding = self._score_grounding(state)
        cost = sum(state.counts[k] for k in ("search", "revisit", "crop")) / max(1, state.max_steps)
        correct = self._judge_correct(state)
        timely = (self.post_gold_search_decay ** state.post_gold_searches
                  if correct and hit_all and state.first_all_gold_seen_turn is not None else 0.0)
        state.scores = {"answer": 1.0 if correct else 0.0, "coverage": cov, "full_coverage": hit_all,
                        "cmp": cmp, "grounding": grounding, "access_cost": cost, "post_gold_search": timely}
        return float((1.0 if correct else 0.0) + self.coverage_w * cov + self.full_coverage_w * hit_all
                     + self.cmp_w * (cmp if cov == 1.0 else 0.0)
                     + self.grounding_w * grounding - self.access_cost_w * cost
                     + self.post_gold_search_w * timely)

    # ------------------------------------------------------------------ info

    def _build_info(self, state: EpisodeState, *, reward: float, action: str) -> dict[str, Any]:
        """Every info carries the same keys (verl-agent collates the ra_* fields of all steps): the per-step extras
        default to STEP_DEFAULTS and are overwritten by _step_one."""
        s = state.scores if state.done else {}
        arrays = [np.asarray(im.convert("RGB")) for im in state.images]
        user_messages = [dict(msg) for msg in state.messages if msg["role"] == "user"]
        gold = state.reference_pages
        return dict(STEP_DEFAULTS) | {
            "ra_hit_pages": tuple(sorted(state.hit_pages)),
            "ra_hit_all_gt": bool(state.done and gold and gold <= set(state.final_pages)),
            "ra_retrieval_reward": s.get("coverage"),
            "won": bool(s.get("answer", 0.0) >= 1.0),
            "task_score": float(reward),
            "data_source": state.data_source,
            "ra_action": action,
            "ra_anchor": state.anchor,
            "ra_gt_pages": tuple(sorted(state.reference_pages)),
            "ra_missing_gt_pages": tuple(sorted(state.reference_pages - state.hit_pages)),
            "ra_final_answer": state.final_answer,
            "ra_final_pages": tuple(state.final_pages),
            "ra_answer_reward": s.get("answer"),
            "ra_coverage": s.get("coverage"),
            "ra_cmp_reward": s.get("cmp"),
            "ra_grounding_reward": s.get("grounding"),
            "ra_access_cost": s.get("access_cost"),
            "ra_budget_exhausted": state.budget_exhausted,
            "ra_forced_answer": state.forced_answer,
            "observation_assistant_prefix": PREFIX if state.forced_answer and not state.done else "",
            "ra_terminated_by": state.terminated_by,
            "ra_seen_pages": tuple(sorted(state.seen_pages)),
            "ra_first_all_gold_seen_turn": state.first_all_gold_seen_turn,
            "ra_post_gold_searches": state.post_gold_searches,
            "ra_post_gold_search_reward": s.get("post_gold_search"),
            "ra_post_gold_search_bonus": self.post_gold_search_w * float(s.get("post_gold_search", 0.0)),
            "ra_counts": dict(state.counts),
            "ra_is_answer": state.done,
            "valid_actions": valid_actions(state),
            "reference_pages": tuple(sorted(state.reference_pages)),
            "retrieved_page_history": tuple(state.retrieved_page_history),
            "observation_image": arrays[0] if arrays else None,
            "observation_chat": [dict(msg) for msg in state.messages],
            "observation_images": arrays,
            "current_user_content": self._observation(state),
            "current_user_images": [],
            "trajectory_user_messages": user_messages,
            "trajectory_user_images": arrays,
        }


# ---------------------------------------------------------------------- helpers

# The per-step fields every info carries (verl-agent's rollout loop copies them into the batch for all steps alike).
STEP_DEFAULTS: dict[str, Any] = {
    "ra_action": "invalid", "ra_step_kind": None, "ra_step_reward": 0.0, "ra_immediate_reward": 0.0,
    "ra_step_group": None, "ra_step_page": "", "ra_retrieved_page_ids": (), "ra_resolved": (),
    "ra_analyse_reward": None, "ra_analyse_page_id": None, "ra_analyse_judge": None,
    "ra_analyse_expected_action": None, "ra_analyse_update_type": None, "ra_analyse_type_match": None,
    "format_error": False, "format_error_detail": None, "edit_error_detail": None,
}


def valid_actions(state: EpisodeState) -> tuple[str, ...]:
    if state.done:
        return ()
    if state.forced_answer:
        return ("answer",)
    if state.view is None or state.view.zoom is not None:
        return ("search", "revisit", "answer")
    return ("search", "revisit", "crop", "answer")


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    return Image.fromarray(np.asarray(image)).convert("RGB")


def format_page_label(deck_name: str | None, page: int) -> str:
    label = f"page_{page:02d}"
    return f"{deck_name}/{label}" if deck_name else label


def normalize_reference_pages(value: Any, *, deck_name: str | None) -> set[str]:
    pages: set[str] = set()
    if value is None:
        return pages
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return pages
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                return normalize_reference_pages(json.loads(stripped), deck_name=deck_name)
            except Exception:  # noqa: BLE001
                pass
        return {stripped}
    for page in value:
        try:
            pages.add(format_page_label(deck_name, int(page)))
        except (TypeError, ValueError):
            pages.add(str(page))
    return pages


def normalize_keyfacts(value: Any, *, deck_name: str | None) -> list[dict[str, Any]]:
    """[{page: int|label, bbox_2d: [x1,y1,x2,y2] (page px frame)}] -> labels + float boxes."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:  # noqa: BLE001
            return []
    out: list[dict[str, Any]] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox_2d") or item.get("bbox")
        page = item.get("page")
        if page is None or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            label = format_page_label(deck_name, int(page))
        except (TypeError, ValueError):
            label = str(page)
        out.append({"page": label, "bbox_2d": [float(v) for v in box]})
    return out


def default_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict[str, Any] | None = None) -> float:
    return 1.0 if str(solution_str).strip().lower() == str(ground_truth).strip().lower() else -1.0


_RETRIEVERS: dict[str, tuple[dict[str, Any], Any]] = {}     # retriever config -> (per-corpus retrievers, single retriever)
_RETRIEVERS_LOCK = threading.Lock()


def _build_retrievers(cfg: Any) -> tuple[dict[str, Any], Any]:
    """(corpus -> retriever, None) for index_paths, else ({}, the single index_path retriever). The corpora share the
    first retriever's ColQwen query encoder."""
    from src.retriever import ColQwenRetriever, make_retriever

    model_path = str(_cfg_get(cfg, "retriever_path", "Qwen/Qwen3-VL-Embedding-8B"))
    kw = dict(device=str(_cfg_get(cfg, "device", "cuda")), dtype=str(_cfg_get(cfg, "dtype", "bfloat16")),
              attn_implementation=_cfg_get(cfg, "attn_implementation", "flash_attention_2"),
              bank_device=_cfg_get(cfg, "bank_device", None))
    paths = parse_index_paths(_cfg_get(cfg, "index_paths", None))
    if not paths:
        return {}, make_retriever(model_path, str(_cfg_get(cfg, "index_path", "data/indexes/slidevqa")), **kw)
    out: dict[str, Any] = {}
    first = None
    for corpus, path in paths.items():
        if first is None:
            first = make_retriever(model_path, path, **kw)
            out[corpus] = first
        else:
            if not isinstance(first, ColQwenRetriever):
                raise ValueError("several index_paths are supported for ColQwen retrievers only")
            r = ColQwenRetriever(model_path=model_path, index_path=path, load_model=False, **kw)
            r.model, r.proc = first.model, first.proc
            out[corpus] = r
    return out, None


def parse_index_paths(value: Any) -> dict[str, str]:
    """"train=/a,val=/b" (or a mapping) -> {"train": "/a", "val": "/b"}; empty -> {}."""
    if not value:
        return {}
    if hasattr(value, "items"):
        return {str(k): str(v) for k, v in value.items()}
    out = {}
    for part in str(value).split(","):
        if part.strip():
            corpus, _, path = part.partition("=")
            if not path:
                raise ValueError(f"index_paths entry {part!r} is not corpus=path")
            out[corpus.strip()] = path.strip()
    return out


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict) or hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def build_slidevqa_envs(seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True, env_config: Any | None = None):
    return SlideVQAMultiProcessEnv(seed=seed, env_num=env_num, group_n=group_n, is_train=is_train, env_config=env_config)
