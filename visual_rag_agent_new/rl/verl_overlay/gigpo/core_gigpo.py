# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from collections import defaultdict, Counter
from verl import DataProto
import uuid

from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any


"""
Core functions to implement the GiGPO algorithm (https://arxiv.org/abs/2505.10978).
The function implemented in this file should be used by trainer with different distributed strategies to implement GiGPO.
"""

# ---------------------------------------------------------- #
# --------------- General Functions of GiGPO --------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            
def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    """
    Compute discounted returns for each trajectory. (Eq. 5 in the paper)
    
    Args:
        batch (DataProto): Input batch.
        gamma (float): Discount factor.
    
    Returns:
        torch.Tensor: Discounted returns.
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        # Calculate returns
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        
        # Calculate returns from the end to the start
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return
        
        # Store the results
        returns_by_traj[uid] = traj_returns
    
    # Recombine the returns into the original batch order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
        all_returns[i] = returns_by_traj[uid][idx_in_traj]
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns

# ---------------------------------------------------------- #
# ---------------- Core Functions of GiGPO ----------------- #
# ---------------------------------------------------------- #

def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   ):
    """
    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode relative advantages (Eq. 3 in the paper).
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
    
    # Anchor state grouping (Eq. 6 in the paper).
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

    # Compute step relative advantages (Eq. 7 in the paper).
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    # Compute joint advantages (Eq. 8 in the paper).
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores


def compute_ra_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                       response_mask: torch.Tensor,
                                       anchor_obs: np.array,
                                       next_anchor_obs: np.array,
                                       index: np.array,
                                       traj_index: np.array,
                                       is_answer: Any = None,
                                       step_kind: Any = None,
                                       action: Any = None,
                                       analyse_reward: Any = None,
                                       analyse_page_id: Any = None,
                                       analyse_judge: Any = None,
                                       gt_pages: Any = None,
                                       step_index: Any = None,
                                       step_uid: Any = None,
                                       epsilon: float = 1e-6,
                                       gamma: float = 0.95,
                                       step_advantage_w: float = 0.3,
                                       analyse_reward_w: float = 0.2,
                                       env_rewards: Any = None,
                                       action_spans: Any = None,
                                       mode: str = "mean_std_norm",
                                       min_group_size: int = 2,
                                       first_visit_only: bool = True,
                                       ):
    """
    Compute Reference-Anchored GRPO advantages.

    RA-GRPO keeps the ordinary episode-level GRPO advantage (which alone carries
    the terminal answer signal and is independent of gamma), then adds a
    GiGPO-style step-relative advantage. Each scored step uses only its own
    immediate reward (run with gamma=0, so no future return is folded in):
    - SEARCH: first-hit reference-page transition is +1; repeated GT hits and
      non-GT retrievals are -1. Scored in its own anchor group.
    - UPDATE_GRAPH: direct 0/0.5/1 reward on its own tokens (0 wrong type,
      0.5 right type, 1 right type + correct next step). Scored in its own
      anchor group so its scale does not mix with SEARCH.
    - ANSWER: enters only the episode-level advantage; not a step group.
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    episode_advantages = episode_norm_reward(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        epsilon=epsilon,
        remove_std=remove_std,
        compute_mean_std_cross_steps=False,
    )

    if action_spans is not None:
        step_advantages = compute_ra_trajectory_span_advantages(
            action_spans=action_spans,
            index=index,
            response_mask=response_mask,
            gamma=gamma,
            analyse_reward_w=analyse_reward_w,
            epsilon=epsilon,
            remove_std=remove_std,
            min_group_size=min_group_size,
            first_visit_only=first_visit_only,
            device=response_mask.device,
        ) * step_advantage_w
        scores = episode_advantages + step_advantages
        return scores, scores

    step_advantage_values = compute_ra_non_answer_step_advantages(
        anchor_obs=anchor_obs,
        next_anchor_obs=next_anchor_obs,
        index=index,
        traj_index=traj_index,
        is_answer=is_answer,
        step_kind=step_kind,
        action=action,
        analyse_reward=analyse_reward,
        analyse_page_id=analyse_page_id,
        analyse_judge=analyse_judge,
        gt_pages=gt_pages,
        env_rewards=env_rewards,
        analyse_reward_w=analyse_reward_w,
        gamma=gamma,
        device=response_mask.device,
        epsilon=epsilon,
        remove_std=remove_std,
        min_group_size=min_group_size,
        first_visit_only=first_visit_only,
    )
    response_length = response_mask.shape[-1]
    step_advantages = (
        step_advantage_w * step_advantage_values
    ).unsqueeze(-1).tile([1, response_length]) * response_mask

    scores = episode_advantages + step_advantages
    return scores, scores


def compute_ra_trajectory_span_advantages(
    *,
    action_spans: Any,
    index: np.array,
    response_mask: torch.Tensor,
    gamma: float = 0.95,
    analyse_reward_w: float = 1.0,
    device: Any = None,
    epsilon: float = 1e-6,
    remove_std: bool = False,
    min_group_size: int = 2,
    first_visit_only: bool = False,
) -> torch.Tensor:
    """Scatter SEARCH step advantages onto full-trajectory response tokens."""
    n_traj, response_length = response_mask.shape
    flat_rows: list[dict[str, Any]] = []
    for traj_row in range(n_traj):
        spans = action_spans[traj_row] if action_spans is not None else []
        if spans is None:
            spans = []
        for local_idx, span in enumerate(list(spans)):
            flat_rows.append(
                {
                    "traj_row": traj_row,
                    "local_idx": local_idx,
                    "span": span,
                }
            )

    token_advantages = torch.zeros((n_traj, response_length), dtype=torch.float32, device=device)
    if not flat_rows:
        return token_advantages

    anchors = np.array([row["span"].get("anchor_obs") for row in flat_rows], dtype=object)
    next_anchors = np.array([row["span"].get("next_anchor_obs") for row in flat_rows], dtype=object)
    span_index = np.array([index[row["traj_row"]] for row in flat_rows], dtype=object)
    traj_index = np.array([f"{row['traj_row']}" for row in flat_rows], dtype=object)
    is_answer = np.array([bool(row["span"].get("is_answer", False)) for row in flat_rows], dtype=object)
    actions = np.array([row["span"].get("action") for row in flat_rows], dtype=object)
    step_kind = np.array([row["span"].get("step_kind") for row in flat_rows], dtype=object)
    analyse_rewards = np.array([row["span"].get("analyse_reward") for row in flat_rows], dtype=object)
    analyse_page_ids = np.array([row["span"].get("analyse_page_id") for row in flat_rows], dtype=object)
    env_rewards = np.array([row["span"].get("env_reward", 0.0) for row in flat_rows], dtype=object)
    step_index = np.array([row["span"].get("step_index", row["local_idx"]) for row in flat_rows], dtype=object)
    step_uid = np.array([row["span"].get("step_uid", f"{row['traj_row']}:{row['local_idx']}") for row in flat_rows], dtype=object)

    span_advantages = compute_ra_non_answer_step_advantages(
        anchor_obs=anchors,
        next_anchor_obs=next_anchors,
        index=span_index,
        traj_index=traj_index,
        is_answer=is_answer,
        step_kind=step_kind,
        action=actions,
        analyse_reward=analyse_rewards,
        analyse_page_id=analyse_page_ids,
        env_rewards=env_rewards,
        analyse_reward_w=analyse_reward_w,
        step_index=step_index,
        step_uid=step_uid,
        gamma=gamma,
        device=device,
        epsilon=epsilon,
        remove_std=remove_std,
        min_group_size=min_group_size,
        first_visit_only=first_visit_only,
    )

    for row, advantage in zip(flat_rows, span_advantages):
        if advantage.item() == 0.0:
            continue
        span = row["span"]
        start = max(0, int(span.get("token_start", 0)))
        end = min(response_length, int(span.get("token_end", 0)))
        if end <= start:
            continue
        traj_row = row["traj_row"]
        token_advantages[traj_row, start:end] = advantage
    return token_advantages * response_mask


def compute_ra_non_answer_step_advantages(anchor_obs: np.array,
                                          next_anchor_obs: np.array,
                                          index: np.array,
                                          traj_index: np.array,
                                          is_answer: Any = None,
                                          step_kind: Any = None,
                                          action: Any = None,
                                          analyse_reward: Any = None,
                                          analyse_page_id: Any = None,
                                          analyse_judge: Any = None,
                                          gt_pages: Any = None,
                                          env_rewards: Any = None,
                                          analyse_reward_w: float = 1.0,
                                          step_index: Any = None,
                                          step_uid: Any = None,
                                          gamma: float = 0.95,
                                          device: Any = None,
                                          epsilon: float = 1e-6,
                                          remove_std: bool = False,
                                          min_group_size: int = 2,
                                          first_visit_only: bool = False,
                                          ) -> torch.Tensor:
    """Return scalar GiGPO-style step advantages for SEARCH and UPDATE_GRAPH.

    Group key is `(prompt_group, anchor_state, action)`, so SEARCH and legal
    UPDATE_GRAPH (analyse) rows are scored in separate same-action groups and
    their reward scales ({-1,+1} vs {0,0.5,1}) do not mix. ANSWER and
    format-error rows are not scored here (ANSWER enters the episode-level
    advantage instead). With gamma=0 each row uses only its own immediate
    reward; with gamma>0 downstream returns fold into earlier rows.
    """
    if len(anchor_obs) != len(next_anchor_obs):
        raise ValueError("anchor_obs and next_anchor_obs must have the same length")
    if len(anchor_obs) != len(index) or len(anchor_obs) != len(traj_index):
        raise ValueError("anchor/index/traj_index arrays must have the same length")
    if min_group_size < 1:
        raise ValueError("min_group_size must be >= 1")

    n_items = len(anchor_obs)
    anchor_sets = [normalize_ra_anchor(anchor) for anchor in anchor_obs]
    next_anchor_sets = [normalize_ra_anchor(anchor) for anchor in next_anchor_obs]
    if is_answer is None:
        answer_flags = np.zeros(n_items, dtype=bool)
    else:
        answer_flags = np.array(is_answer, dtype=bool)
    action_values = normalize_string_array(action, n_items)
    kind_values = normalize_step_kind_flags(step_kind, n_items)
    page_values = normalize_string_array(analyse_page_id, n_items)
    # Per-row grouping observation. SEARCH rows compare per anchor only (there is
    # no current image at the decide step). UPDATE_GRAPH rows additionally compare
    # per observed page, so two analyse decisions only share a group when they
    # face the SAME accumulated hits AND the SAME image (same expected action).
    group_obs = np.array(
        [page_values[i] if action_values[i] == "update_graph" else "" for i in range(n_items)],
        dtype=object,
    )
    # SEARCH rows and legal UPDATE_GRAPH (analyse) rows each get a direct step
    # advantage; ANSWER and format-error rows do not. UPDATE_GRAPH carries its
    # own 0/0.5/1 immediate reward (see analyse_reward_values below), normalized
    # within its own action-type group so its scale does not mix with SEARCH.
    active = np.array(
        [
            (not answer_flags[i])
            and (
                action_values[i] == "search"
                or (action_values[i] == "update_graph" and kind_values[i] == "analyse")
            )
            for i in range(n_items)
        ],
        dtype=bool,
    )

    search_rewards = np.array(
        [
            1.0 if action_values[i] == "search" and next_anchor_sets[i] > anchor_sets[i]
            else -1.0 if action_values[i] == "search"
            else 0.0
            for i in range(n_items)
        ],
        dtype=np.float32,
    )
    env_reward_values = normalize_float_array(env_rewards, n_items)
    analyse_reward_values = normalize_float_array(analyse_reward, n_items) * float(analyse_reward_w)
    analyse_reward_values = np.where(kind_values == "analyse", analyse_reward_values, 0.0).astype(np.float32)
    immediate_rewards = search_rewards + analyse_reward_values + env_reward_values
    discounted_returns = discounted_returns_by_trajectory(
        immediate_rewards=immediate_rewards,
        traj_index=traj_index,
        step_index=step_index,
        step_uid=step_uid,
        gamma=gamma,
    )

    if first_visit_only:
        first_visit_rows: dict[tuple[Any, Any], int] = {}
        step_order = np.arange(n_items) if step_index is None else np.array(step_index)
        step_ids = np.array([str(i) for i in range(n_items)], dtype=object) if step_uid is None else np.array(step_uid, dtype=object)
        for row in range(n_items):
            if not active[row]:
                continue
            key = (traj_index[row], tuple(sorted(anchor_sets[row])), action_values[row], group_obs[row])
            current = first_visit_rows.get(key)
            if current is None:
                first_visit_rows[key] = row
                continue
            current_order = (step_order[current], str(step_ids[current]))
            row_order = (step_order[row], str(step_ids[row]))
            if row_order < current_order:
                first_visit_rows[key] = row
        first_visit_mask = np.zeros(n_items, dtype=bool)
        for row in first_visit_rows.values():
            first_visit_mask[row] = True
        active = np.logical_and(active, first_visit_mask)

    # Group key includes the action type so SEARCH ({-1,+1}) and UPDATE_GRAPH
    # ({0,0.5,1}) rows never normalize together, plus group_obs so UPDATE_GRAPH
    # rows are compared only against same-page decisions.
    groups: dict[tuple[Any, tuple[str, ...], str, Any], list[int]] = defaultdict(list)
    for row in range(n_items):
        if active[row]:
            groups[(index[row], tuple(sorted(anchor_sets[row])), action_values[row], group_obs[row])].append(row)

    advantages = torch.zeros(n_items, dtype=torch.float32, device=device)
    returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32, device=device)
    with torch.no_grad():
        for rows in groups.values():
            if len(rows) < min_group_size:
                continue
            values = returns_tensor[rows]
            mean = values.mean()
            if remove_std:
                normalized = values - mean
            else:
                std = values.std(unbiased=False)
                normalized = (values - mean) / (std + epsilon)
            for local_idx, row in enumerate(rows):
                advantages[row] = normalized[local_idx]
    return advantages


def compute_ra_anchor_step_advantages(anchor_obs: np.array,
                                      next_anchor_obs: np.array,
                                      index: np.array,
                                      traj_index: np.array,
                                      is_answer: Any = None,
                                      step_kind: Any = None,
                                      action: Any = None,
                                      step_index: Any = None,
                                      step_uid: Any = None,
                                      device: Any = None,
                                      epsilon: float = 1e-6,
                                      remove_std: bool = False,
                                      min_group_size: int = 2,
                                      first_visit_only: bool = True,
                                      ) -> torch.Tensor:
    """
    Return one scalar RA-GRPO anchor advantage per generated Decide-search step.

    The active comparison set is first visits to the same `(prompt_group,
    anchor_state)`. Non-first visits, non-search steps, answer actions, and
    undersized groups receive zero step advantage.
    """
    if len(anchor_obs) != len(next_anchor_obs):
        raise ValueError("anchor_obs and next_anchor_obs must have the same length")
    if len(anchor_obs) != len(index) or len(anchor_obs) != len(traj_index):
        raise ValueError("anchor/index/traj_index arrays must have the same length")
    if min_group_size < 1:
        raise ValueError("min_group_size must be >= 1")

    n_items = len(anchor_obs)
    anchor_sets = [normalize_ra_anchor(anchor) for anchor in anchor_obs]
    next_anchor_sets = [normalize_ra_anchor(anchor) for anchor in next_anchor_obs]
    step_rewards = torch.tensor(
        [1.0 if next_anchor_sets[i] > anchor_sets[i] else 0.0 for i in range(n_items)],
        dtype=torch.float32,
        device=device,
    )

    if is_answer is None:
        answer_flags = np.zeros(n_items, dtype=bool)
    else:
        answer_flags = np.array(is_answer, dtype=bool)
    kind_flags = normalize_step_kind_flags(step_kind, n_items)
    action_values = normalize_string_array(action, n_items)

    if step_index is None:
        # Preserve backward compatibility for tests or callers that have not
        # been through the rollout collector. In normal training this is an
        # explicit trajectory step number, so balancing/reordering is harmless.
        step_order = np.arange(n_items)
    else:
        step_order = np.array(step_index)

    if step_uid is None:
        step_ids = np.array([str(i) for i in range(n_items)], dtype=object)
    else:
        step_ids = np.array(step_uid, dtype=object)

    if action is None:
        search_flags = np.logical_not(answer_flags)
    else:
        search_flags = np.array([value == "search" for value in action_values], dtype=bool)
    active = np.logical_and.reduce((kind_flags == "decide", search_flags, np.logical_not(answer_flags)))
    if first_visit_only:
        first_visit_rows: dict[tuple[Any, Any], int] = {}
        for row in range(n_items):
            if not active[row]:
                continue
            key = (traj_index[row], tuple(sorted(anchor_sets[row])))
            current = first_visit_rows.get(key)
            if current is None:
                first_visit_rows[key] = row
                continue
            current_order = (step_order[current], str(step_ids[current]))
            row_order = (step_order[row], str(step_ids[row]))
            if row_order < current_order:
                first_visit_rows[key] = row

        first_visit_mask = np.zeros(n_items, dtype=bool)
        seen_step_ids: set[str] = set()
        for row in first_visit_rows.values():
            row_step_id = str(step_ids[row])
            if row_step_id in seen_step_ids:
                continue
            first_visit_mask[row] = True
            seen_step_ids.add(row_step_id)
        active = np.logical_and(active, first_visit_mask)

    groups: dict[tuple[Any, tuple[str, ...]], list[int]] = defaultdict(list)
    for row in range(n_items):
        if not active[row]:
            continue
        group_key = (index[row], tuple(sorted(anchor_sets[row])))
        groups[group_key].append(row)

    advantages = torch.zeros(n_items, dtype=torch.float32, device=device)
    with torch.no_grad():
        for rows in groups.values():
            if len(rows) < min_group_size:
                continue
            values = step_rewards[rows]
            mean = values.mean()
            if remove_std:
                normalized = values - mean
            else:
                std = values.std(unbiased=False)
                normalized = (values - mean) / (std + epsilon)
            for local_idx, row in enumerate(rows):
                advantages[row] = normalized[local_idx]
    return advantages


def compute_ra_step_advantages(*args, **kwargs) -> torch.Tensor:
    """Backward-compatible alias for the Decide anchor component."""
    return compute_ra_anchor_step_advantages(*args, **kwargs)


def normalize_float_array(value: Any, n_items: int, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(n_items, default, dtype=np.float32)
    array = np.array(value, dtype=object)
    values = np.full(n_items, default, dtype=np.float32)
    for i in range(min(n_items, len(array))):
        if array[i] is None:
            continue
        try:
            values[i] = float(array[i])
        except (TypeError, ValueError):
            values[i] = default
    return values


def discounted_returns_by_trajectory(
    *,
    immediate_rewards: np.ndarray,
    traj_index: np.ndarray,
    step_index: Any = None,
    step_uid: Any = None,
    gamma: float = 1.0,
) -> np.ndarray:
    traj_ids = np.array(traj_index, dtype=object)
    if step_index is None:
        step_order = np.arange(len(immediate_rewards))
    else:
        step_order = np.array(step_index)
    if step_uid is None:
        step_ids = np.array([str(i) for i in range(len(immediate_rewards))], dtype=object)
    else:
        step_ids = np.array(step_uid, dtype=object)

    returns = np.zeros(len(immediate_rewards), dtype=np.float32)
    for traj in np.unique(traj_ids):
        rows = np.where(traj_ids == traj)[0].tolist()
        rows.sort(key=lambda row: (step_order[row], str(step_ids[row])))
        running_return = 0.0
        for row in reversed(rows):
            running_return = float(immediate_rewards[row]) + gamma * running_return
            returns[row] = running_return
    return returns


def compute_ra_analyse_rewards(step_kind: Any = None,
                               analyse_reward: Any = None,
                               analyse_page_id: Any = None,
                               analyse_judge: Any = None,
                               gt_pages: Any = None,
                               n_items: Any = None,
                               device: Any = None,
                               ) -> torch.Tensor:
    """Return pointwise Analyse judge rewards, with no group normalization."""
    if n_items is None:
        n_items = infer_step_count(step_kind, analyse_reward, analyse_page_id, analyse_judge, gt_pages)
    kind_flags = normalize_step_kind_flags(step_kind, n_items)

    if analyse_reward is not None:
        raw_rewards = np.array(analyse_reward, dtype=object)
        values = [
            float(raw_rewards[i]) if kind_flags[i] == "analyse" and raw_rewards[i] is not None else 0.0
            for i in range(n_items)
        ]
    else:
        pages = normalize_string_array(analyse_page_id, n_items)
        judges = normalize_string_array(analyse_judge, n_items)
        gt_sets = normalize_gt_pages_array(gt_pages, n_items)
        values = [
            analyse_judge_reward(pages[i], judges[i], gt_sets[i]) if kind_flags[i] == "analyse" else 0.0
            for i in range(n_items)
        ]
    return torch.tensor(values, dtype=torch.float32, device=device)


def analyse_judge_reward(page_id: str, judge: str, gt_pages: frozenset[str]) -> float:
    is_gt = page_id in gt_pages
    if is_gt:
        if judge == "yes":
            return 1.0
        if judge == "partial":
            return 0.5
        return -1.0
    if judge == "no":
        return 1.0
    if judge == "partial":
        return -0.5
    return -1.0


def infer_step_count(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            return len(value)
        if isinstance(value, (list, tuple)):
            return len(value)
    return 0


def normalize_step_kind_flags(step_kind: Any, n_items: int) -> np.ndarray:
    if step_kind is None:
        return np.array(["decide"] * n_items, dtype=object)
    return normalize_string_array(step_kind, n_items, default="decide")


def normalize_string_array(value: Any, n_items: int, default: str = "") -> np.ndarray:
    if value is None:
        return np.array([default] * n_items, dtype=object)
    array = np.array(value, dtype=object)
    if array.ndim == 0:
        array = np.array([array.item()] * n_items, dtype=object)
    if len(array) != n_items:
        raise ValueError(f"expected {n_items} items, got {len(array)}")
    return np.array([str(item or default).strip().lower() for item in array], dtype=object)


def normalize_gt_pages_array(value: Any, n_items: int) -> list[frozenset[str]]:
    if value is None:
        return [frozenset() for _ in range(n_items)]
    array = np.array(value, dtype=object)
    if array.ndim == 0:
        array = np.array([array.item()] * n_items, dtype=object)
    if len(array) != n_items:
        raise ValueError(f"expected {n_items} items, got {len(array)}")
    return [normalize_ra_anchor(item) for item in array]


def normalize_ra_anchor(anchor: Any) -> frozenset[str]:
    """Normalize a reference-page anchor representation to a frozenset."""
    if anchor is None:
        return frozenset()
    if isinstance(anchor, frozenset):
        return frozenset(str(item) for item in anchor)
    if isinstance(anchor, set):
        return frozenset(str(item) for item in anchor)
    if isinstance(anchor, np.ndarray):
        return normalize_ra_anchor(anchor.tolist())
    if isinstance(anchor, dict):
        for key in ("hit_pages", "reference_pages", "pages", "anchor"):
            if key in anchor:
                return normalize_ra_anchor(anchor[key])
        return frozenset(str(value) for _, value in sorted(anchor.items()))
    if isinstance(anchor, (list, tuple)):
        return frozenset(str(item) for item in anchor)
    if isinstance(anchor, str):
        stripped = anchor.strip()
        if not stripped:
            return frozenset()
        if stripped.startswith("[") and stripped.endswith("]"):
            import json
            try:
                return normalize_ra_anchor(json.loads(stripped))
            except Exception:
                pass
        if "|" in stripped:
            return frozenset(part.strip() for part in stripped.split("|") if part.strip())
        return frozenset([stripped])
    return frozenset([str(anchor)])


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False, similarity_thresh: float = 0.95, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    enable_similarity : bool
        Whether to enable similarity-based step-level grouping (default: False)
    similarity_thresh : float
        Threshold for similarity to consider two observations as identical (default: 1.0, meaning exact match)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    if enable_similarity:
        assert similarity_thresh > 0.0 and similarity_thresh < 1.0, "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size: List[int] = []
    # Process each unique index
    for idx in unique_indices:
        if not enable_similarity:
            # Get all observations for this index using np.where
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            
            # Create clusters for identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
            
            # Assign unique step_group_uid to each cluster
            for obs, original_indices in clusters.items():
                # Generate a UUID for this cluster
                uid = str(uuid.uuid4())
                
                # Assign the same step_group_uid to all elements in this cluster
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]

            # Dynamically maintain clusters: [{rep: str, locs: List[int]} ...]
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, locs):
                 # Try to place into an existing cluster
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                # If no matching cluster, create a new one
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            # Assign a UUID to each cluster
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages
