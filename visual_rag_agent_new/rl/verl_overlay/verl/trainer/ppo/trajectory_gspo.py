"""Whole-trajectory GSPO over unchanged, separately conditioned turn tensors.

Rows remain the exact inputs used at sampling time. Only grouping and the loss
reduction change: all action tokens in a trajectory share one importance ratio.
"""
from collections import OrderedDict

import numpy as np
import torch


def trajectory_groups(ids, valid=None):
    groups = OrderedDict()
    if valid is None:
        valid = [True] * len(ids)
    for row, (uid, keep) in enumerate(zip(ids, valid)):
        if keep:
            groups.setdefault(str(uid), []).append(row)
    return groups


def pack_trajectory_shards(data, world_size, minibatch_size, question_balanced=False):
    """Keep every trajectory on one DP rank; padding is dispatch-only, never trained."""
    groups = trajectory_groups(data.non_tensor_batch['traj_uid'])
    assert len(groups) % minibatch_size == 0
    assert minibatch_size % world_size == 0
    target = len(groups) // world_size
    lengths = data.batch['attention_mask'].sum(-1).tolist()
    items = list(groups.items())
    costs = [sum(lengths[i] for i in rows) for _, rows in items]
    ranks, loads = [[] for _ in range(world_size)], [0] * world_size
    if question_balanced:
        # One complete trajectory from EACH question per optimizer step.
        # Balance token cost only within each fixed global minibatch, keeping
        # each rank's minibatch sequence intact after distributed dispatch.
        question_groups = OrderedDict()
        for idx, (_, rows) in enumerate(items):
            question_ids = {str(data.non_tensor_batch['uid'][i]) for i in rows}
            assert len(question_ids) == 1, 'A trajectory spans different questions'
            question_groups.setdefault(next(iter(question_ids)), []).append(idx)
        assert len(question_groups) == minibatch_size
        repeats = len(items) // minibatch_size
        assert all(len(v) == repeats for v in question_groups.values())
        per_rank = minibatch_size // world_size
        for repeat in range(repeats):
            mini = [indices[repeat] for indices in question_groups.values()]
            sizes = [0] * world_size
            for idx in sorted(mini, key=lambda i: (-costs[i], i)):
                rank = min((r for r in range(world_size) if sizes[r] < per_rank), key=lambda r: (loads[r], r))
                ranks[rank].append(idx); sizes[rank] += 1; loads[rank] += costs[idx]
            assert sizes == [per_rank] * world_size
        for repeat in range(repeats):
            ids = [str(data.non_tensor_batch['uid'][items[i][1][0]])
                   for rank in ranks for i in rank[repeat*per_rank:(repeat+1)*per_rank]]
            assert len(ids) == len(set(ids)) == minibatch_size
        ordered_ranks = ranks
    else:
        for idx in sorted(range(len(items)), key=lambda i: (-costs[i], i)):
            rank = min((r for r in range(world_size) if len(ranks[r]) < target), key=lambda r: (loads[r], r))
            ranks[rank].append(idx)
            loads[rank] += costs[idx]
        ordered_ranks = [sorted(part) for part in ranks]
    partitions = [[row for i in part for row in items[i][1]] for part in ordered_ranks]
    width = max(map(len, partitions))
    indices, valid = [], []
    for part in partitions:
        indices.extend(part + [part[0]] * (width - len(part)))
        valid.extend([True] * len(part) + [False] * (width - len(part)))
    packed = data.select_idxs(np.asarray(indices))
    packed.meta_info = dict(data.meta_info)
    packed.batch['trajectory_row_valid'] = torch.tensor(valid, dtype=torch.bool, device=packed.batch.device)
    packed.meta_info['trajectory_count_per_rank'] = target
    packed.meta_info['question_balanced_minibatches'] = bool(question_balanced)
    return packed, {'trajectory/count': len(groups), 'trajectory/dispatch_padding_rows': len(indices)-len(data),
                    'trajectory/minibatch_size': minibatch_size,
                    'trajectory/optimizer_steps_per_round': len(groups)//minibatch_size,
                    'trajectory/question_balanced_minibatches': float(question_balanced)}


def trajectory_reduce(values, mask, row_to_trajectory, token_lengths, trajectory_count):
    """Equal trajectory weight, normalized over its action tokens across all turns."""
    valid = row_to_trajectory >= 0
    safe = row_to_trajectory.clamp_min(0)
    weights = valid.to(values.dtype) / token_lengths[safe].to(values.dtype) / trajectory_count
    return ((values * mask).sum(-1) * weights).sum()


def trajectory_statistics(log_probs, old_log_probs, masks, row_to_trajectory, trajectory_count):
    acc_dtype = torch.float64 if log_probs.dtype == torch.float64 else torch.float32
    lengths = torch.zeros(trajectory_count, dtype=acc_dtype, device=log_probs.device)
    log_ratios = torch.zeros_like(lengths)
    valid = row_to_trajectory >= 0
    ids = row_to_trajectory[valid]
    lengths.scatter_add_(0, ids, masks[valid].to(acc_dtype).sum(-1))
    delta = ((log_probs[valid].to(acc_dtype) - old_log_probs[valid].to(acc_dtype)) * masks[valid].to(acc_dtype)).sum(-1)
    log_ratios.scatter_add_(0, ids, delta)
    assert bool(torch.all(lengths > 0)), 'Empty trajectory action mask'
    return (log_ratios / lengths).detach(), lengths.detach()


def trajectory_policy_loss(log_probs, old_log_probs, advantages, mask, row_to_trajectory,
                           log_ratios, token_lengths, trajectory_count, clip_low, clip_high):
    """GSPO's detached ratio surrogate, shared across every turn of a trajectory."""
    safe = row_to_trajectory.clamp_min(0)
    log_ratio = log_probs - log_probs.detach() + log_ratios[safe].unsqueeze(-1)
    ratio = log_ratio.clamp(max=10.0).exp()
    raw = -advantages * ratio
    clipped = -advantages * ratio.clamp(1-clip_low, 1+clip_high)
    pg_loss = trajectory_reduce(torch.maximum(raw, clipped), mask, row_to_trajectory, token_lengths, trajectory_count)
    clipfrac = trajectory_reduce((clipped > raw).float(), mask, row_to_trajectory, token_lengths, trajectory_count)
    kl = trajectory_reduce(old_log_probs-log_probs, mask, row_to_trajectory, token_lengths, trajectory_count)
    return pg_loss, clipfrac, kl


def validate_trajectory_advantages(advantages, mask, row_to_trajectory, trajectory_count):
    for idx in range(trajectory_count):
        values = advantages[row_to_trajectory == idx][mask[row_to_trajectory == idx].bool()]
        assert values.numel() and torch.isfinite(values).all()
        assert torch.allclose(values, values[0].expand_as(values), atol=1e-6, rtol=1e-5), \
            'Trajectory GSPO requires a single outcome advantage per trajectory'
