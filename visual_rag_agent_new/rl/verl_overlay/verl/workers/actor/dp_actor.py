# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
import time
import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist
import numpy as np
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.trainer.ppo.trajectory_gspo import (trajectory_groups, trajectory_statistics, trajectory_policy_loss,
                                             trajectory_reduce, validate_trajectory_advantages)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _synchronize_micro_batches(batch, multi_modal_inputs, batches, indices, max_token_len):
    """Match FSDP forward branches as well as call counts across DP ranks.

    A text-only padding row skips the sharded vision blocks and desynchronizes
    NCCL collectives. Every microbatch must therefore have an image when any
    rank has one. Extra image rows are dispatch-only, with index -1, and are
    discarded from log-probs or assigned zero trajectory loss.
    """
    device = batch['input_ids'].device
    lengths = batch['attention_mask'].sum(-1)
    image_rows = [i for i, item in enumerate(multi_modal_inputs) if item] if multi_modal_inputs is not None else []
    flags = torch.tensor([bool(image_rows)], device=device, dtype=torch.long)
    if dist.is_initialized():
        flags_all = [torch.zeros_like(flags) for _ in range(dist.get_world_size())]
        dist.all_gather(flags_all, flags)
        has_images = [bool(x.item()) for x in flags_all]
    else:
        has_images = [bool(image_rows)]
    need_image = any(has_images)
    shortest = min(image_rows, key=lambda i: int(lengths[i])) if image_rows else int(lengths.argmin())
    template = dict(batch[shortest:shortest + 1])
    template_mm = multi_modal_inputs[shortest:shortest + 1] if multi_modal_inputs is not None else None
    if need_image and not all(has_images):
        # Rare all-text rank: borrow one real image row only for zero-loss
        # dispatch. All ranks participate in this one explicit broadcast.
        source = has_images.index(True)
        payload = [None]
        if dist.get_rank() == source:
            payload[0] = ({k: v.detach().cpu() for k, v in template.items()},
                          [{k: v.detach().cpu() for k, v in item.items()} for item in template_mm])
        dist.broadcast_object_list(payload, src=source, device=device)
        if not image_rows:
            template = {k: v.to(device) for k, v in payload[0][0].items()}
            template_mm = np.empty(1, dtype=object)
            template_mm[0] = {k: v.to(device) for k, v in payload[0][1][0].items()}
    # Copies only: never change the real row from which this template came.
    if 'trajectory_index' in template:
        template['trajectory_index'] = torch.full_like(template['trajectory_index'], -1)
    if 'trajectory_action_mask' in template:
        template['trajectory_action_mask'] = torch.zeros_like(template['trajectory_action_mask'])
    template_cost = int(template['attention_mask'].sum())
    result, result_indices = [], []
    for mb, idx in zip(batches, indices):
        idx = list(idx)
        needs_dummy = need_image and not any(bool(multi_modal_inputs[i]) for i in idx) if multi_modal_inputs is not None else need_image
        parts = [(mb, idx)]
        if needs_dummy:
            budget = max_token_len - template_cost
            assert budget >= int(mb['attention_mask'].sum(-1).max()), 'Image dispatch template exceeds token budget'
            pieces, piece_indices = rearrange_micro_batches(mb, budget, same_micro_num_in_dp=False)
            parts = [(piece, [idx[i] for i in pi]) for piece, pi in zip(pieces, piece_indices)]
        for part, real_idx in parts:
            item = dict(part)
            mm = multi_modal_inputs[real_idx] if multi_modal_inputs is not None else None
            if needs_dummy:
                item = {k: torch.cat((v, template[k]), dim=0) for k, v in item.items()}
                mm = np.concatenate((mm, template_mm)) if mm is not None else template_mm
                real_idx = real_idx + [-1]
            if mm is not None:
                item['multi_modal_inputs'] = mm
            result.append(item); result_indices.append(real_idx)
    count = torch.tensor(len(result), device=device, dtype=torch.long)
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
    for _ in range(int(count) - len(result)):
        item = dict(template)
        if template_mm is not None:
            item['multi_modal_inputs'] = template_mm
        result.append(item); result_indices.append([-1])
    return result, result_indices


def _dynamic_micro_batches(batch, multi_modal_inputs, max_token_len, synchronize=False):
    micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
    if synchronize:
        return _synchronize_micro_batches(batch, multi_modal_inputs, micro_batches, indices, max_token_len)
    # Preserve the legacy per-turn update path; its loss does not understand
    # zero-loss dispatch rows. Whole-trajectory updates use the helper above.
    if multi_modal_inputs is not None:
        micro_batches = [{**mb, 'multi_modal_inputs': multi_modal_inputs[idx]} for mb, idx in zip(micro_batches, indices)]
    return micro_batches, indices


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            # text-only rows carry {} (no page in view); concatenate the rows that have images
            rows = [inputs for inputs in micro_batch["multi_modal_inputs"] if inputs]
            for key in (rows[0].keys() if rows else ()):
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in rows], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if use_dynamic_bsz:
            # split using dynamic bsz; image rows keep their multi_modal_inputs (same indices)
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = _dynamic_micro_batches(batch, data.non_tensor_batch.get("multi_modal_inputs"), max_token_len, synchronize=True)
        elif has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            keep = torch.tensor([idx >= 0 for idx in indices], dtype=torch.bool, device=log_probs.device)
            log_probs = log_probs[keep]
            if entropys is not None:
                entropys = entropys[keep]
            indices = [idx for idx in indices if idx >= 0]
            assert sorted(indices) == list(range(len(data))), 'Missing or duplicate real log-prob rows'
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if entropys is not None:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    def _trajectory_micro_batches(self, mini):
        if self.config.use_dynamic_bsz:
            batches, indices = rearrange_micro_batches(
                mini.batch, self.config.ppo_max_token_len_per_gpu, same_micro_num_in_dp=False)
        else:
            size = self.config.ppo_micro_batch_size_per_gpu
            indices = [list(range(i, min(i + size, len(mini)))) for i in range(0, len(mini), size)]
            batches = [mini.batch[idx] for idx in indices]
        batches, _ = _synchronize_micro_batches(
            mini.batch, mini.non_tensor_batch.get('multi_modal_inputs'), batches, indices,
            self.config.ppo_max_token_len_per_gpu)
        return batches

    def _update_trajectory_policy(self, data):
        assert not self.use_ulysses_sp, 'Grouped trajectory replay currently supports SP=1 only'
        assert self.config.ppo_epochs == 1, 'Preserve one PPO epoch for this experiment'
        # Two-pass replay needs deterministic forwards at fixed weights.
        for module in self.actor_module.modules():
            if isinstance(module, nn.Dropout):
                assert module.p == 0, 'Nonzero dropout invalidates two-pass trajectory ratios'
            if hasattr(module, 'attention_dropout'):
                assert module.attention_dropout == 0, 'Attention dropout must be zero'
        self.actor_module.train()
        temperature = data.meta_info['temperature']
        valid = data.batch['trajectory_row_valid'].cpu().tolist()
        groups = trajectory_groups(data.non_tensor_batch['traj_uid'], valid)
        assert len(groups) == data.meta_info['trajectory_count_per_rank']
        size = self.config.ppo_mini_batch_size  # FSDP worker has divided global trajectories by DP size.
        assert len(groups) % size == 0
        items = list(groups.values())
        metrics = {}
        clip_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else self.config.clip_ratio
        clip_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else self.config.clip_ratio
        for start in range(0, len(items), size):
            selected = items[start:start + size]
            if data.meta_info.get('question_balanced_minibatches', False):
                assert len({str(data.non_tensor_batch['uid'][rows[0]]) for rows in selected}) == size
            rows = [row for group in selected for row in group]
            mini = data.select_idxs(np.asarray(rows))
            mini.batch['trajectory_index'] = torch.tensor(
                [i for i, group in enumerate(selected) for _ in group],
                device=mini.batch['input_ids'].device, dtype=torch.long)
            if data.meta_info.get('multi_turn', False):
                mask = mini.batch['loss_mask'][:, -mini.batch['responses'].size(-1):]
            else:
                mask = mini.batch['attention_mask'][:, -mini.batch['responses'].size(-1):]
            mini.batch['trajectory_action_mask'] = mask
            validate_trajectory_advantages(mini.batch['advantages'], mask, mini.batch['trajectory_index'], size)
            micros = self._trajectory_micro_batches(mini)
            current, old, masks, ids = [], [], [], []
            with torch.no_grad():
                for mb in micros:
                    _, lp = self._forward_micro_batch(mb, temperature=temperature, calculate_entropy=False)
                    current.append(lp)
                    old.append(mb['old_log_probs'])
                    masks.append(mb['trajectory_action_mask'])
                    ids.append(mb['trajectory_index'])
                ratios, lengths = trajectory_statistics(torch.cat(current), torch.cat(old), torch.cat(masks), torch.cat(ids), size)
            del current, old, masks, ids, lp
            # As in FSDPWorker.compute_log_prob, release the root's unsharded
            # inference buffer before switching back to gradient-enabled replay.
            if isinstance(self.actor_module, FSDP) and dist.is_initialized() and dist.get_world_size() > 1:
                if self.actor_module._handle is not None:
                    self.actor_module._handle.reshard(True)
            self.actor_optimizer.zero_grad(set_to_none=True)
            totals = dict.fromkeys(('actor/pg_loss', 'actor/pg_clipfrac', 'actor/ppo_kl', 'actor/kl_loss',
                                   'actor/pg_clipfrac_lower'), 0.0)
            if self.config.entropy_coeff != 0:
                totals['actor/entropy_loss'] = 0.0
            for mb in micros:
                entropy, lp = self._forward_micro_batch(mb, temperature=temperature,
                                                      calculate_entropy=self.config.entropy_coeff != 0)
                mask, ids = mb['trajectory_action_mask'], mb['trajectory_index']
                pg, clipped, ppo_kl = trajectory_policy_loss(
                    lp, mb['old_log_probs'], mb['advantages'], mask, ids,
                    ratios, lengths, size, clip_low, clip_high)
                loss = pg
                if self.config.use_kl_loss:
                    kld = kl_penalty(logprob=lp, ref_logprob=mb['ref_log_prob'], kl_penalty=self.config.kl_loss_type)
                    kl = trajectory_reduce(kld, mask, ids, lengths, size)
                    loss = loss + self.config.kl_loss_coef * kl
                    totals['actor/kl_loss'] += kl.detach().item()
                if self.config.entropy_coeff != 0:
                    ent = trajectory_reduce(entropy, mask, ids, lengths, size)
                    loss = loss - self.config.entropy_coeff * ent
                    totals['actor/entropy_loss'] += ent.detach().item()
                # Normalized by all action tokens in each complete trajectory;
                # there is no additional turn-count or microbatch-size factor.
                loss.backward()
                totals['actor/pg_loss'] += pg.detach().item()
                totals['actor/pg_clipfrac'] += clipped.detach().item()
                totals['actor/ppo_kl'] += ppo_kl.detach().item()
            totals['actor/grad_norm'] = self._optimizer_step().detach().item()
            totals['actor/kl_coef'] = self.config.kl_loss_coef
            totals['trajectory/local_minibatch_count'] = size
            totals['trajectory/micro_forwards_per_pass'] = len(micros)
            # Keep each minibatch visible instead of hiding staleness behind a round mean.
            # Diagnostics only: no additional sampling, backward pass or optimizer step.
            prefix = f"trajectory/mini_{start // size + 1:02d}/"
            for key in ('pg_loss', 'pg_clipfrac', 'ppo_kl', 'grad_norm'):
                totals[prefix + key] = totals['actor/' + key]
            totals[prefix + 'preupdate_mean_abs_log_ratio'] = ratios.abs().mean().item()
            totals[prefix + 'mean_action_tokens'] = lengths.mean().item()
            totals[prefix + 'updates_since_collection'] = start // size
            append_to_dict(metrics, totals)
        self.actor_optimizer.zero_grad(set_to_none=True)
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        if self.config.policy_loss.get("loss_mode") == "gspo_trajectory":
            return self._update_trajectory_policy(data)
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    if has_multi_modal_inputs:
                        micro_batches, _ = _dynamic_micro_batches(
                            mini_batch.batch, mini_batch.non_tensor_batch["multi_modal_inputs"], max_token_len)
                    else:
                        micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                elif has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    elif isinstance(data, dict):  # dynamic-bsz image micro batch (see _dynamic_micro_batches)
                        dev = get_torch_device().current_device()
                        data = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (responses.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
