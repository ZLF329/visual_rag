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

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import Any, List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import os


def _active_subset_on() -> bool:
    return os.environ.get("ROLLOUT_ACTIVE_SUBSET", "1") != "0"


def _ra_anchor_set(value: Any) -> set:
    if value is None:
        return set()
    if isinstance(value, (set, frozenset)):
        return set(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return set(value.tolist() if isinstance(value, np.ndarray) else value)
    return {value}


def _finalize_update_graph_step_rewards(action_spans: list) -> None:
    """Apply the one-step lookahead for UPDATE_GRAPH rewards (0 / 0.5 / 1).

    Each analyse (UPDATE_GRAPH) span carries a pre-lookahead reward of 0.0
    (wrong type) or 0.5 (right type), emitted by the env. Here, with the full
    ordered trajectory available, a right-type step is upgraded to 1.0 when the
    immediately following step is itself correct: a SEARCH that hit a new
    ground-truth page, an UPDATE_GRAPH with the expected type, or a correct
    ANSWER. The reward logic lives in the slidevqa env (single source of truth).
    """
    from agent_system.environments.env_package.slidevqa.envs import (
        update_graph_next_step_correct,
        update_graph_step_reward,
    )

    spans = list(action_spans)
    for idx, span in enumerate(spans):
        if span.get("step_kind") != "analyse":
            continue
        type_match = bool(span.get("analyse_type_match"))
        next_span = spans[idx + 1] if idx + 1 < len(spans) else None
        if next_span is None:
            next_correct = False
        else:
            answer_reward = next_span.get("answer_reward")
            next_correct = update_graph_next_step_correct(
                next_action=next_span.get("action"),
                next_hit=_ra_anchor_set(next_span.get("next_anchor_obs"))
                > _ra_anchor_set(next_span.get("anchor_obs")),
                next_type_matches=bool(next_span.get("analyse_type_match")),
                next_answer_correct=answer_reward is not None and float(answer_reward) >= 1.0,
            )
        span["analyse_reward"] = update_graph_step_reward(
            type_matches=type_match, next_step_correct=next_correct
        )


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def _slidevqa_loss_granularity(self) -> str:
        env_config = getattr(self.config, "env", None)
        slidevqa_config = getattr(env_config, "slidevqa", {}) if env_config is not None else {}
        if isinstance(slidevqa_config, dict):
            return str(slidevqa_config.get("loss_granularity", "step")).lower()
        if hasattr(slidevqa_config, "get"):
            return str(slidevqa_config.get("loss_granularity", "step")).lower()
        return str(getattr(slidevqa_config, "loss_granularity", "step")).lower()

    def _trajectory_response_length(self) -> int:
        env_config = getattr(self.config, "env", None)
        slidevqa_config = getattr(env_config, "slidevqa", {}) if env_config is not None else {}
        default = int(self.config.data.max_response_length)
        if isinstance(slidevqa_config, dict):
            return int(slidevqa_config.get("trajectory_response_length", default))
        if hasattr(slidevqa_config, "get"):
            return int(slidevqa_config.get("trajectory_response_length", default))
        return int(getattr(slidevqa_config, "trajectory_response_length", default))

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None
        # visual_rag_agent_new (DESIGN.md §11.31): the env gives each turn's full context (system / question /
        # previous response / observation + memory) and every image in view; generate from exactly that.
        env_chat = obs['chat'][item] if obs.get('chat') is not None else None
        env_images = list(obs['images'][item]) if env_chat is not None else []
        if env_chat is not None:
            is_multi_modal = len(env_images) > 0
            n_slots = sum(str(m.get("content", "")).count("<image>") for m in env_chat)
            if n_slots != len(env_images):
                raise ValueError(f"env chat has {n_slots} <image> slots but {len(env_images)} images")
        # A new-protocol turn with no page in view (e.g. a search that found no new page) is text-only, but it is
        # batched with image turns: it keeps the Qwen-VL row layout (mrope position ids, empty image inputs).
        text_only_vlm_turn = (env_chat is not None and not is_multi_modal and self.processor is not None
                              and hasattr(self.processor, "image_processor"))

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        if env_chat is not None:
            chat = np.array([{"role": str(m["role"]), "content": str(m["content"])} for m in env_chat])
        
        # Apply chat template
        prefixes = obs.get("assistant_prefix")
        prefix = prefixes[item] if prefixes is not None else ""
        prompt_with_chat_template = _render_observation_prompt(
            self.tokenizer, chat, apply_chat_template_kwargs, prefix)
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(im) for im in env_images] if env_chat is not None
                                            else [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
            image_grid_thw = None
            if text_only_vlm_turn:
                # same keys as the image rows it is batched with (an empty image list is not sent to vLLM)
                row_dict['multi_modal_data'] = {'image': []}
                row_dict['multi_modal_inputs'] = {}

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal or text_only_vlm_turn:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        if prefix and raw_prompt_ids != input_ids[0, attention_mask[0].bool()].tolist():
            raise RuntimeError("Forced-answer sampling and actor prompt tokens differ")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        granularity = self._slidevqa_loss_granularity()
        if granularity not in ('step', 'trajectory', 'trajectory_grouped'):
            raise ValueError(f'Unknown SlideVQA loss granularity: {granularity}')
        if granularity == 'trajectory':
            return self.gather_trajectory_rollout_data(
                total_batch_list=total_batch_list,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                success=success,
                traj_uid=traj_uid,
                tool_callings=tool_callings,
            )

        batch_size = len(total_batch_list)
        if granularity == 'trajectory_grouped':
            for i, steps in enumerate(total_batch_list):
                active = [s for s in steps if s['active_masks']]
                assert len(active) == int(episode_lengths[i]) > 0, 'Incomplete trajectory turn list'
                assert [int(s['step_index']) for s in active] == list(range(len(active))), 'Missing or duplicated trajectory turn'
                assert bool(active[-1]['ra_is_answer']), 'Trajectory has not reached a terminal state'
                for step in active:
                    r = step['responses'].shape[-1]
                    assert torch.equal(step['input_ids'][-r:], step['responses']), 'Response/input alignment mismatch'
                    assert step['attention_mask'][-r:].sum() > 0, 'Empty action response'

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def gather_trajectory_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        success_rate = {key: np.mean(value) for key, value in success.items()}
        rows: list[dict[str, Any]] = []
        for bs, trajectory_steps in enumerate(total_batch_list):
            active_steps = [step for step in trajectory_steps if step.get("active_masks")]
            if not active_steps:
                continue
            row = self._build_trajectory_training_row(
                steps=active_steps,
                episode_reward=float(episode_rewards[bs]),
                episode_length=float(episode_lengths[bs]),
                traj_uid=str(traj_uid[bs]),
                tool_calling=float(tool_callings[bs]),
                success_rate=success_rate,
            )
            rows.append(row)
        return DataProto.from_single_dict(data=collate_fn(rows))

    def _build_trajectory_training_row(
        self,
        *,
        steps: list[dict[str, Any]],
        episode_reward: float,
        episode_length: float,
        traj_uid: str,
        tool_calling: float,
        success_rate: dict[str, float],
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        user_images: list[Any] = []
        assistant_message_indices: list[int] = []
        assistant_step_rows: list[dict[str, Any]] = []

        for step in steps:
            raw_prompt = step.get("raw_prompt")
            if raw_prompt is not None and len(raw_prompt) > 0:
                user_content = str(raw_prompt[0].get("content", ""))
            else:
                user_content = self.tokenizer.decode(step["prompts"], skip_special_tokens=False)
            messages.append({"role": "user", "content": user_content})
            for image in self._step_prompt_images(step):
                user_images.append(image)

            response_ids = step["responses"]
            prompt_len = step["prompts"].shape[-1]
            valid_response_length = int(step["attention_mask"][prompt_len:].sum().item())
            response_text = self.tokenizer.decode(
                response_ids[:valid_response_length].tolist(),
                skip_special_tokens=True,
            )
            assistant_message_indices.append(len(messages))
            assistant_step_rows.append(step)
            messages.append({"role": "assistant", "content": response_text})

        full_text = self._render_chat(messages, add_generation_prompt=False, images=user_images)
        first_prompt_text = self._render_chat([messages[0]], add_generation_prompt=True, images=self._message_prefix_images(messages, user_images, 1))
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(first_prompt_text, add_special_tokens=False)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("trajectory chat template prefix mismatch")

        loss_mask_full = [0] * len(full_ids)
        action_spans: list[dict[str, Any]] = []
        for assistant_idx, step in zip(assistant_message_indices, assistant_step_rows):
            end_messages = messages[: assistant_idx + 1]
            prefix_images = self._message_prefix_images(messages, user_images, assistant_idx)
            start_text = self._render_chat(messages[:assistant_idx], add_generation_prompt=True, images=prefix_images)
            end_text = self._render_chat(end_messages, add_generation_prompt=False, images=prefix_images)
            start = len(self.tokenizer.encode(start_text, add_special_tokens=False))
            end = len(self.tokenizer.encode(end_text, add_special_tokens=False))
            for pos in range(start, min(end, len(loss_mask_full))):
                loss_mask_full[pos] = 1
            response_start = max(0, start - len(prompt_ids))
            response_end = max(0, end - len(prompt_ids))
            action_spans.append(
                {
                    "action": step.get("ra_action"),
                    "anchor_obs": step.get("anchor_obs"),
                    "next_anchor_obs": step.get("next_anchor_obs"),
                    "is_answer": step.get("ra_is_answer", False),
                    "step_kind": step.get("ra_step_kind"),
                    "analyse_reward": step.get("ra_analyse_reward"),
                    "analyse_page_id": step.get("ra_analyse_page_id"),
                    "analyse_judge": step.get("ra_analyse_judge"),
                    "analyse_expected_action": step.get("ra_analyse_expected_action"),
                    "analyse_update_type": step.get("ra_analyse_update_type"),
                    "analyse_type_match": step.get("ra_analyse_type_match"),
                    "env_reward": step.get("rewards", 0.0),
                    "final_answer": step.get("ra_final_answer"),
                    "answer_reward": step.get("ra_answer_reward"),
                    "retrieval_reward": step.get("ra_retrieval_reward"),
                    "hit_all_gt": step.get("ra_hit_all_gt"),
                    "missing_gt_pages": step.get("ra_missing_gt_pages"),
                    "step_index": step.get("step_index"),
                    "step_uid": step.get("step_uid"),
                    "token_start": response_start,
                    "token_end": response_end,
                }
            )

        _finalize_update_graph_step_rewards(action_spans)

        final_step = assistant_step_rows[-1] if assistant_step_rows else steps[-1]
        max_prompt_length = int(self.config.data.max_prompt_length)
        max_response_length = self._trajectory_response_length()
        response_ids = full_ids[len(prompt_ids):]
        response_loss_mask = loss_mask_full[len(prompt_ids):]
        if len(prompt_ids) > max_prompt_length:
            raise RuntimeError(f"Trajectory prompt length {len(prompt_ids)} exceeds max_prompt_length={max_prompt_length}.")
        if len(response_ids) > max_response_length:
            raise RuntimeError(
                f"Trajectory response length {len(response_ids)} exceeds trajectory_response_length={max_response_length}."
            )

        prompt_pad = max_prompt_length - len(prompt_ids)
        response_pad = max_response_length - len(response_ids)
        pad_id = self.tokenizer.pad_token_id
        prompt_tensor = torch.tensor([pad_id] * prompt_pad + prompt_ids, dtype=torch.long)
        response_tensor = torch.tensor(response_ids + [pad_id] * response_pad, dtype=torch.long)
        input_ids = torch.cat([prompt_tensor, response_tensor], dim=0)
        attention_mask = torch.tensor([0] * prompt_pad + [1] * len(prompt_ids) + [1] * len(response_ids) + [0] * response_pad, dtype=torch.long)
        loss_mask = torch.tensor([0] * max_prompt_length + response_loss_mask + [0] * response_pad, dtype=torch.long)

        row: dict[str, Any] = {
            "prompts": prompt_tensor,
            "responses": response_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "raw_prompt_ids": prompt_ids,
            "uid": steps[0].get("uid"),
            "traj_uid": traj_uid,
            "index": steps[0].get("index"),
            "data_source": steps[0].get("data_source"),
            "episode_rewards": episode_reward,
            "episode_lengths": episode_length,
            "tool_callings": tool_calling,
            "ra_action_spans": action_spans,
            "ra_final_answer": final_step.get("ra_final_answer") or "",
            "ra_answer_reward": final_step.get("ra_answer_reward"),
            "ra_retrieval_reward": final_step.get("ra_retrieval_reward"),
            "ra_hit_all_gt": final_step.get("ra_hit_all_gt"),
            "ra_missing_gt_pages": final_step.get("ra_missing_gt_pages"),
            "ra_gt_pages": final_step.get("ra_gt_pages"),
            "retrieved_page_history": final_step.get("retrieved_page_history"),
        }
        for key, value in success_rate.items():
            row[key] = value

        if user_images:
            row["multi_modal_data"] = {"image": user_images}
            image_inputs = self.processor.image_processor(user_images, return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]
            row["multi_modal_inputs"] = {key: val for key, val in image_inputs.items()}
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
            valid_mask = attention_mask.bool()
            text_position_ids = torch.ones((1, len(input_ids)), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            row["position_ids"] = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            row["position_ids"] = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]

        return row

    def _step_prompt_images(self, step: dict[str, Any]) -> list[Any]:
        data = step.get("multi_modal_data")
        if isinstance(data, dict) and data.get("image") is not None:
            return list(data["image"])
        return []

    def _message_prefix_images(self, messages: list[dict[str, str]], images: list[Any], n_messages: int) -> list[Any]:
        image_count = 0
        for message in messages[:n_messages]:
            if message.get("role") == "user":
                image_count += str(message.get("content", "")).count("<image>")
        return images[:image_count]

    def _render_chat(self, messages: list[dict[str, str]], *, add_generation_prompt: bool, images: list[Any]) -> str:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **self.config.data.get("apply_chat_template_kwargs", {}),
        )
        if not images:
            return rendered
        image_inputs = self.processor.image_processor(images, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        index = 0
        merge_length = self.processor.image_processor.merge_size**2
        while "<image>" in rendered:
            if index >= len(image_grid_thw):
                raise ValueError("trajectory chat contains more <image> placeholders than supplied images")
            grid = image_grid_thw[index]
            rendered = rendered.replace(
                "<image>",
                "<|vision_start|>" + "<|placeholder|>" * (grid.prod() // merge_length) + "<|vision_end|>",
                1,
            )
            index += 1
        if index != len(image_grid_thw):
            raise ValueError("trajectory chat supplied images that were not referenced by <image> placeholders")
        return rendered.replace("<|placeholder|>", self.processor.image_token)

    def _generate_active(self, batch_input: DataProto, active_masks: np.ndarray, actor_rollout_wg) -> DataProto:
        """Generate only for the trajectories still running (ROLLOUT_ACTIVE_SUBSET=1, the default; 0 = every row, the
        old lockstep). A finished trajectory gets an empty response (all pad, attention 0): its row is recorded with
        active_masks=False and dropped by gather_rollout_data, and the env ignores a finished state's action, so the
        collected data is the lockstep loop's without the wasted generation (DESIGN.md §11.34)."""
        world = actor_rollout_wg.world_size
        idx = np.nonzero(active_masks)[0]
        if not _active_subset_on() or len(idx) == len(active_masks):
            padded, pad_size = pad_dataproto_to_divisor(batch_input, world)
            return unpad_dataproto(actor_rollout_wg.generate_sequences(padded), pad_size=pad_size)
        sub = batch_input.select_idxs(idx)
        sub.meta_info = batch_input.meta_info
        padded, pad_size = pad_dataproto_to_divisor(sub, world)
        out = unpad_dataproto(actor_rollout_wg.generate_sequences(padded), pad_size=pad_size)
        return self._scatter_active_output(batch_input, out, idx)

    def _scatter_active_output(self, batch_input: DataProto, out: DataProto, idx: np.ndarray) -> DataProto:
        """Full-batch generation output: the generated rows at idx, an empty response for every other row (same
        layout as vLLM's: prompts | responses, attention 0 on the response, positions continuing the prompt's)."""
        B = len(batch_input)
        rows = torch.as_tensor(idx, dtype=torch.long)
        prompts = batch_input.batch["input_ids"]
        prompt_mask = batch_input.batch["attention_mask"]
        pos = batch_input.batch["position_ids"]
        R = out.batch["responses"].size(1)
        responses = torch.full((B, R), self.tokenizer.pad_token_id, dtype=out.batch["responses"].dtype)
        delta = torch.arange(1, R + 1, dtype=pos.dtype).view(*([1] * (pos.dim() - 1)), R)
        filler = {
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=-1),
            "attention_mask": torch.cat([prompt_mask, torch.zeros((B, R), dtype=prompt_mask.dtype)], dim=-1),
            "position_ids": torch.cat([pos, pos[..., -1:] + delta], dim=-1),
        }
        tensors = {}
        for key, val in out.batch.items():
            base = filler.get(key)
            if base is None:  # e.g. rollout_log_probs
                base = torch.full((B,) + tuple(val.shape[1:]), -1 if val.is_floating_point() else 0, dtype=val.dtype)
            if tuple(base.shape[1:]) != tuple(val.shape[1:]):
                raise ValueError(f"active-subset filler for {key} has shape {tuple(base.shape)}, output {tuple(val.shape)}")
            base = base.clone().to(device=val.device, dtype=val.dtype)
            base[rows.to(val.device)] = val
            tensors[key] = base
        non_tensors = {}
        for key, val in out.non_tensor_batch.items():
            full = batch_input.non_tensor_batch.get(key)
            arr = np.array(list(full), dtype=object) if full is not None and len(full) == B else np.empty(B, dtype=object)
            for j, i in enumerate(idx):
                arr[i] = val[j]
            non_tensors[key] = arr
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=out.meta_info)

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.get('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # The automatic question search already consumed one environment turn.
        # G gets one extra text-only call after the ordinary decision budget.
        _sv = self.config.env.get("slidevqa", {}) if hasattr(self.config.env, "get") else {}
        _hard_budget = str((_sv or {}).get("mode", "")) == "new"
        _model_turns = _rollout_turn_limit(self.config.env)
        active_counts: list[int] = []
        smoke_forced_gpu_rows = 0
        for _step in range(_model_turns):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = dict(gen_batch.meta_info)
            prefixes = obs.get("assistant_prefix")
            if prefixes is not None and any(prefix for prefix, active in zip(prefixes, active_masks) if active):
                if not all(prefix for prefix, active in zip(prefixes, active_masks) if active):
                    raise RuntimeError("Forced-answer and ordinary turns mixed in one generation batch")
                batch_input.meta_info["forced_answer_max_tokens"] = 96

            # generate for the trajectories still running only (ROLLOUT_ACTIVE_SUBSET, default on)
            batch_output = self._generate_active(batch_input, active_masks, actor_rollout_wg)
            active_counts.append(int(active_masks.sum()))

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            if batch_input.meta_info.get("forced_answer_max_tokens"):
                width = batch.batch["responses"].shape[-1]
                selected_rows = torch.as_tensor(active_masks, device=batch.batch["attention_mask"].device)
                lengths = batch.batch["attention_mask"][selected_rows, -width:].sum(-1)
                if not bool(((lengths > 0) & (lengths <= 96)).all()):
                    raise RuntimeError("Invalid G response mask")
                smoke_forced_gpu_rows += int(lengths.numel())

            next_obs, rewards, dones, infos = envs.step(text_actions)

            batch.non_tensor_batch['step_index'] = np.array([_step for _ in range(batch_size)], dtype=object)
            batch.non_tensor_batch['step_uid'] = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
            next_anchor_obs = next_obs.get('anchor', None)
            if next_anchor_obs is not None:
                if isinstance(next_anchor_obs, torch.Tensor):
                    next_anchor_obs = torch_to_numpy(next_anchor_obs, is_object=True)
                elif isinstance(next_anchor_obs, np.ndarray):
                    next_anchor_obs = next_anchor_obs.astype(object)
                else:
                    next_anchor_obs = np.array(next_anchor_obs, dtype=object)
                batch.non_tensor_batch['next_anchor_obs'] = next_anchor_obs

            for info_key, batch_key in (
                ('ra_step_kind', 'ra_step_kind'),
                ('ra_is_answer', 'ra_is_answer'),
                ('ra_terminated_by', 'ra_terminated_by'),
                ('ra_budget_exhausted', 'ra_budget_exhausted'),
                ('ra_forced_answer', 'ra_forced_answer'),
                ('ra_seen_pages', 'ra_seen_pages'),
                ('ra_first_all_gold_seen_turn', 'ra_first_all_gold_seen_turn'),
                ('ra_post_gold_searches', 'ra_post_gold_searches'),
                ('ra_post_gold_search_reward', 'ra_post_gold_search_reward'),
                ('ra_post_gold_search_bonus', 'ra_post_gold_search_bonus'),
                ('ra_action', 'ra_action'),
                ('ra_retrieved_page_ids', 'ra_retrieved_page_ids'),
                ('ra_hit_pages', 'ra_hit_pages'),
                ('ra_gt_pages', 'ra_gt_pages'),
                ('ra_analyse_reward', 'ra_analyse_reward'),
                ('ra_analyse_page_id', 'ra_analyse_page_id'),
                ('ra_analyse_judge', 'ra_analyse_judge'),
                ('ra_analyse_expected_action', 'ra_analyse_expected_action'),
                ('ra_analyse_update_type', 'ra_analyse_update_type'),
                ('ra_analyse_type_match', 'ra_analyse_type_match'),
                ('ra_final_answer', 'ra_final_answer'),
                ('ra_answer_reward', 'ra_answer_reward'),
                ('ra_retrieval_reward', 'ra_retrieval_reward'),
                ('ra_hit_all_gt', 'ra_hit_all_gt'),
                ('ra_missing_gt_pages', 'ra_missing_gt_pages'),
                ('retrieved_page_history', 'retrieved_page_history'),
            ):
                if any(info_key in info for info in infos):
                    batch.non_tensor_batch[batch_key] = np.array(
                        [info.get(info_key) for info in infos],
                        dtype=object,
                    )

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        if os.environ.get("PREFILL_GPU_SMOKE") == "1":
            import json
            terminal = [infos[int(length) - 1] for infos, length in zip(total_infos, episode_lengths)]
            forced = [info for info in terminal if info.get("ra_forced_answer")]
            for info, value in zip(terminal, episode_rewards):
                if info["ra_terminated_by"] == "answer":
                    correct = float(info["ra_answer_reward"])
                    timely = correct * bool(info["ra_hit_all_gt"]) * (info["ra_first_all_gold_seen_turn"] is not None)
                    timely *= 0.5 ** info["ra_post_gold_searches"]
                    expected = correct + 0.5 * float(info["ra_grounding_reward"]) + 0.5 * timely
                else:
                    expected = -1.0
                if abs(float(value) - expected) > 1e-5:
                    raise RuntimeError(f"Trajectory reward mismatch: {value} != {expected}")
            print("PREFILL_GPU_AUDIT " + json.dumps({
                "trajectories": len(terminal),
                "forced_turns": len(forced),
                "forced_answers": sum(info["ra_terminated_by"] == "answer" for info in forced),
                "forced_correct": sum(info.get("ra_answer_reward") == 1 for info in forced),
                "forced_gpu_rows_checked": smoke_forced_gpu_rows,
                "reward_formula_checked_for_all": True,
            }), flush=True)
        if _hard_budget and not is_done.all():
            raise RuntimeError("Hard-budget rollout ended without terminal rewards for every trajectory")
        print(f"[ROLLOUT_ACTIVE] subset={_active_subset_on()} generated per turn: {','.join(map(str, active_counts))} "
              f"of {batch_size}", flush=True)

        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        return gen_batch_output


def _rollout_turn_limit(env_config):
    """Number of sampled model actions allowed by the environment's hard budget."""
    sv = env_config.get("slidevqa", {}) or {}
    limit = int(env_config.get("max_steps", 10))
    if str(sv.get("mode", "")) == "new":
        limit -= int(sv.get("first_search", "model") == "question")
    if limit <= 0:
        raise ValueError("The budget must leave at least one ordinary model action")
    return limit + int(str(sv.get("mode", "")) == "new" and bool(sv.get("forced_answer_prefill", False)))


def _render_observation_prompt(tokenizer, chat, template_kwargs, prefix):
    """Include the supplied assistant prefix in the context used by sampling and actor replay."""
    if not prefix:
        return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, **template_kwargs)
    rendered = tokenizer.apply_chat_template(
        list(chat), add_generation_prompt=False, continue_final_message=True, tokenize=False, **template_kwargs)
    if not rendered.endswith(prefix):
        raise ValueError("Chat template did not preserve the assistant prefix")
    return rendered
