from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image
from pydantic import BaseModel

from src.memory import Memory
from src.prompt_serialization import build_agent_human_turn
from src.prompts import (
    STRICT_ACTION_TAG_RETRY_SUFFIX,
    STRICT_JSON_RETRY_SUFFIX,
    build_analyse_prompt,
    build_decide_prompt,
    build_evidence_update_prompt,
)
from src.protocol import ParsedTurn, ProtocolError, parse_turn
from src.baseline_prompts import (
    build_image_summary_prompt,
    build_summary_decide_prompt,
)
from src.active_clue_graph import (
    ClueGraph,
    VisualObservation,
    compact_query_history,
    format_graph_state_for_prompt,
    format_observation_for_prompt,
    sufficient_child_answers,
)
from src.schemas import (
    AnalyseResult,
    DecideResult,
    EvidenceState,
    EvidenceUpdateResult,
    GraphDecisionResult,
    ImageSummaryResult,
)


T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    def __init__(self, schema_name: str, outputs: list[str], last_error: Exception) -> None:
        last_output = outputs[-1].replace("\n", "\\n")[:500] if outputs else ""
        super().__init__(
            f"failed to parse {schema_name} after {len(outputs)} attempt(s): "
            f"{last_error}; last_output={last_output!r}"
        )
        self.schema_name = schema_name
        self.outputs = outputs
        self.last_error = last_error



class VLM:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-Instruct-4B",
        device: str = "cuda",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        prompt_mode: str = "chat",
        adapter_path: str | None = None,
        load_model: bool = True,
        provider: str = "qwen",
        api_base_url: str | None = None,
        api_key_env: str = "MIMO_API_KEY",
        api_timeout: float = 180.0,
        api_max_retries: int = 2,
        api_image_quality: int = 85,
        api_extra_body: dict[str, Any] | None = None,
        policy_output_format: str = "tags",
        api_policy_response_format: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.adapter_path = adapter_path
        self.provider = provider
        self.api_base_url = api_base_url
        self.api_key_env = api_key_env
        self.api_timeout = api_timeout
        self.api_max_retries = api_max_retries
        self.api_image_quality = api_image_quality
        self.api_extra_body = api_extra_body or {}
        if policy_output_format not in {"tags", "json"}:
            raise ValueError(f"unsupported policy_output_format: {policy_output_format}")
        self.policy_output_format = policy_output_format
        self.api_policy_response_format = api_policy_response_format
        if prompt_mode not in {"chat", "system_in_user"}:
            raise ValueError(f"unsupported prompt_mode: {prompt_mode}")
        self.prompt_mode = prompt_mode
        self.processor: Any | None = None
        self.model: Any | None = None
        if self.uses_api:
            return
        if load_model:
            self._load_model()

    @property
    def uses_api(self) -> bool:
        return self.provider in {"api", "openai_compatible", "dashscope_api"}

    def decide_next(
        self,
        original_query: str,
        memory: Memory,
        warnings_summary: str | None = None,
    ) -> DecideResult:
        memory_context = warnings_summary or memory.context_for_decide()
        images = self._load_retained_images(memory)
        system, user = build_decide_prompt(
            original_query=original_query,
            memory_context=memory_context,
            retained_visual_evidence_count=len(images),
        )
        return self._generate_structured(
            system=system,
            user=user,
            images=images,
            schema=DecideResult,
            validator=lambda item: item.validate_branch(),
            call_type="decide",
        )

    def generate_turn(
        self,
        *,
        messages: list[dict[str, str]],
        images: list[Image.Image],
        observation_pending: bool,
    ) -> tuple[str, ParsedTurn]:
        """AGv2: generate one policy turn and parse it under the merged-action protocol.
        One format retry with the strict suffix; returns (raw_text, ParsedTurn). Raises
        StructuredOutputError when both attempts are structurally invalid. BoxFormatError
        (malformed bbox payload only) is NOT retried here — the caller owns that soft lane,
        so it propagates on the first attempt."""
        outputs: list[str] = []
        last_error: Exception | None = None
        for attempt in range(2):
            attempt_messages = (
                messages
                if attempt == 0
                else append_suffix_to_last_user_message(messages, STRICT_ACTION_TAG_RETRY_SUFFIX)
            )
            text = self._generate_text_from_messages(
                messages=attempt_messages,
                images=images,
                max_new_tokens=None,
            )
            outputs.append(text)
            try:
                return text, parse_turn(text, observation_pending=observation_pending)
            except ProtocolError as exc:
                last_error = exc
        raise StructuredOutputError(
            ParsedTurn.__name__,
            outputs,
            last_error or RuntimeError("protocol parse error"),
        )

    def verify_answer(
        self,
        *,
        original_query: str,
        images: list,
        page_labels: list | None = None,
        graph_state: str = "",
    ) -> tuple:
        """Final-step verify: re-read the accepted evidence page image(s) together with the
        graph facts and re-derive the answer with explicit reasoning (no recent-turn conv).
        Returns (answer, full_generation, user_text). user_text carries ONE <image> marker per
        evidence image so inference and the SFT row use the IDENTICAL multimodal format."""
        imgs = list(images or [])
        labels_line = ", ".join(page_labels) if page_labels else "(the attached pages)"
        markers = "".join("<image>\n" for _ in imgs)
        user = (
            markers
            + "Agent call type: verify (final answer).\n"
            + f"Root question: {original_query}\n\n"
            + f"The evidence page image(s) above ({labels_line}) are what you used. Below is the "
            + "graph state you built (the facts you committed, each with its bbox_2d).\n\n"
            + f"Graph state:\n{graph_state or '(empty)'}\n\n"
            + "Re-read the attached evidence page(s) CAREFULLY and cross-check them against the graph "
            + "facts, then answer the root question by EXPLICIT step-by-step reasoning over what you "
            + "actually SEE:\n"
            + "- comparison: state BOTH values and which is larger/smaller;\n"
            + "- count: enumerate the items and count them;\n"
            + "- lookup: quote the exact value/text from the page.\n"
            + "Do NOT just repeat a previously stored answer -- re-derive it from the evidence; if the "
            + "evidence contradicts a stored fact, trust the evidence.\n\n"
            + "Write your reasoning, then on the LAST line write exactly:\n"
            + "FINAL ANSWER: <your final answer>"
        )
        text = self._generate_text(
            system="",
            user=user,
            images=imgs,
            max_new_tokens=512,
            call_type="verify",
        )
        return _extract_final_answer(text), text, user

    def analyse(
        self,
        image: Image.Image,
        original_query: str,
        search_query: str | None = None,
    ) -> AnalyseResult:
        system, user = build_analyse_prompt(
            original_query=original_query,
            search_query=search_query,
        )
        result = self._generate_structured(
            system=system,
            user=user,
            images=[image],
            schema=AnalyseResult,
            validator=lambda item: item.validate_branch(),
            call_type="analyse",
        )
        return result

    def update_evidence_state(
        self,
        *,
        image: Image.Image,
        original_query: str,
        search_query: str,
        page_summary: str,
        previous_evidence_state: EvidenceState,
    ) -> EvidenceUpdateResult:
        system, user = build_evidence_update_prompt(
            original_question=original_query,
            search_query=search_query,
            page_summary=page_summary,
            evidence_state_json=json.dumps(
                {
                    "answer_relevant_facts": list(previous_evidence_state.answer_relevant_facts),
                    "missing_requirements": list(previous_evidence_state.missing_requirements),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        return self._generate_structured(
            system=system,
            user=user,
            images=[image],
            schema=EvidenceUpdateResult,
            validator=lambda item: item.validate_branch(),
            max_new_tokens=800,
            call_type="evidence_update",
        )

    def direct_answer(
        self,
        *,
        original_query: str,
        images: list[Image.Image],
        page_labels: list[str],
    ) -> str:
        labels = "\n".join(
            f"  [{idx}] {label}" for idx, label in enumerate(page_labels, start=1)
        )
        system = (
            "Answer the visual document question using only the attached page images. "
            "If the evidence is insufficient, say so briefly."
        )
        user = (
            f"Original question: {original_query}\n\n"
            f"Retrieved pages:\n{labels or '  None'}\n\n"
            "Final answer:"
        )
        return self._generate_text(system=system, user=user, images=images).strip()

    def summary_baseline_decide(
        self,
        *,
        original_query: str,
        summaries_context: str,
        recent_images: list[Image.Image],
        recent_image_labels: str,
    ) -> DecideResult:
        system, user = build_summary_decide_prompt(
            original_query=original_query,
            summaries_context=summaries_context,
            recent_image_labels=recent_image_labels,
        )
        return self._generate_structured(
            system=system,
            user=user,
            images=recent_images,
            schema=DecideResult,
            validator=lambda item: item.validate_branch(),
        )

    def summary_baseline_summarise_image(
        self,
        *,
        image: Image.Image,
        original_query: str,
        search_query: str,
        summaries_context: str,
        page_label: str,
    ) -> ImageSummaryResult:
        system, user = build_image_summary_prompt(
            original_query=original_query,
            search_query=search_query,
            summaries_context=summaries_context,
            page_label=page_label,
        )
        return self._generate_structured(
            system=system,
            user=user,
            images=[image],
            schema=ImageSummaryResult,
            validator=lambda item: item.validate_branch(),
        )

    def _load_model(self) -> None:
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None

        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            AutoModelForVision2Seq = None

        dtype = getattr(torch, self.dtype, torch.bfloat16)

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        model_classes = [
            cls
            for cls in (AutoModelForImageTextToText, AutoModelForVision2Seq)
            if cls is not None
        ]
        if not model_classes:
            raise RuntimeError("no compatible transformers vision-language model class found")

        last_error: Exception | None = None
        for model_class in model_classes:
            try:
                kwargs: dict[str, Any] = {
                    "torch_dtype": dtype,
                    "trust_remote_code": True,
                }
                if self.attn_implementation:
                    kwargs["attn_implementation"] = self.attn_implementation
                if self.device == "cuda":
                    kwargs["device_map"] = "auto"
                self.model = model_class.from_pretrained(self.model_path, **kwargs)
                if self.adapter_path:
                    try:
                        from peft import PeftModel
                    except ImportError as exc:
                        raise RuntimeError("peft is required to load a LoRA adapter for eval") from exc
                    self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
                if self.device != "cuda":
                    self.model.to(self.device)
                self.model.eval()
                return
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"failed to load VLM {self.model_path}: {last_error}")

    def _generate_structured(
        self,
        *,
        system: str,
        user: str,
        images: list[Image.Image],
        schema: type[T],
        validator: Any,
        max_new_tokens: int | None = None,
        call_type: str | None = None,
    ) -> T:
        outputs: list[str] = []
        last_error: Exception | None = None
        for attempt in range(2):
            attempt_user = user if attempt == 0 else user + STRICT_JSON_RETRY_SUFFIX
            text = self._generate_text(
                system=system,
                user=attempt_user,
                images=images,
                max_new_tokens=max_new_tokens,
                call_type=call_type,
            )
            outputs.append(text)
            try:
                payload = extract_json_object(text)
                item = schema.model_validate(payload)
                validator(item)
                return item
            except Exception as exc:
                last_error = exc
        raise StructuredOutputError(schema.__name__, outputs, last_error or RuntimeError("parse error"))

    def _generate_text(
        self,
        *,
        system: str,
        user: str,
        images: list[Image.Image],
        max_new_tokens: int | None = None,
        call_type: str | None = None,
    ) -> str:
        if self.uses_api:
            return self._generate_text_api(
                system=system,
                user=user,
                images=images,
                max_new_tokens=max_new_tokens,
                call_type=call_type,
            )

        if self.model is None or self.processor is None:
            raise RuntimeError(
                "VLM model is not loaded. Instantiate with load_model=True on a GPU machine."
            )

        import torch

        inputs = self._prepare_model_inputs(
            system=system,
            user=user,
            images=images,
            call_type=call_type,
            add_generation_prompt=True,
        )

        do_sample = self.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0]

    def _generate_text_from_messages(
        self,
        *,
        messages: list[dict[str, str]],
        images: list[Image.Image],
        max_new_tokens: int | None = None,
    ) -> str:
        if self.uses_api:
            return self._generate_text_api_from_messages(
                messages=messages,
                images=images,
                max_new_tokens=max_new_tokens,
            )

        if self.model is None or self.processor is None:
            raise RuntimeError(
                "VLM model is not loaded. Instantiate with load_model=True on a GPU machine."
            )

        import torch

        inputs = self._prepare_model_inputs_from_messages(
            messages=messages,
            images=images,
            add_generation_prompt=True,
        )
        do_sample = self.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0]

    def _generate_text_api_from_messages(
        self,
        *,
        messages: list[dict[str, str]],
        images: list[Image.Image],
        max_new_tokens: int | None,
    ) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is missing for API VLM provider")
        base_url = (self.api_base_url or os.environ.get("MIMO_BASE_URL") or "").rstrip("/")
        if not base_url:
            raise RuntimeError("api_base_url or MIMO_BASE_URL is required for API VLM provider")
        payload: dict[str, Any] = {
            "model": self.model_path,
            "messages": build_api_messages_from_chat(messages, images, quality=self.api_image_quality),
            "temperature": self.temperature,
            "max_tokens": max_new_tokens or self.max_tokens,
        }
        payload.update(self.api_extra_body)
        if self.policy_output_format == "json" and self.api_policy_response_format:
            payload.setdefault("response_format", {"type": self.api_policy_response_format})
        return chat_completion_text(
            base_url=base_url,
            api_key=api_key,
            payload=payload,
            timeout=self.api_timeout,
            max_retries=self.api_max_retries,
        )

    def _generate_text_api(
        self,
        *,
        system: str,
        user: str,
        images: list[Image.Image],
        max_new_tokens: int | None,
        call_type: str | None,
    ) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is missing for API VLM provider")
        base_url = (self.api_base_url or os.environ.get("MIMO_BASE_URL") or "").rstrip("/")
        if not base_url:
            raise RuntimeError("api_base_url or MIMO_BASE_URL is required for API VLM provider")

        messages = self._prepare_api_messages(
            system=system,
            user=user,
            images=images,
            call_type=call_type,
        )
        payload: dict[str, Any] = {
            "model": self.model_path,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_new_tokens or self.max_tokens,
        }
        payload.update(self.api_extra_body)
        if call_type == "policy" and self.policy_output_format == "json":
            if self.api_policy_response_format:
                payload.setdefault(
                    "response_format",
                    {"type": self.api_policy_response_format},
                )
        return chat_completion_text(
            base_url=base_url,
            api_key=api_key,
            payload=payload,
            timeout=self.api_timeout,
            max_retries=self.api_max_retries,
        )

    def _prepare_api_messages(
        self,
        *,
        system: str,
        user: str,
        images: list[Image.Image],
        call_type: str | None,
    ) -> list[dict[str, Any]]:
        user_text = user
        messages: list[dict[str, Any]] = []
        if self.prompt_mode == "system_in_user":
            if not (
                call_type == "analyse"
                and not system.strip()
                and user_text.lstrip().startswith("Agent call type: analyse")
            ):
                user_text = build_agent_human_turn(system, user, call_type=call_type)
        elif system:
            messages.append({"role": "system", "content": system})

        if not images:
            messages.append({"role": "user", "content": user_text})
            return messages

        content: list[dict[str, Any]] = []
        remaining_images = list(images)
        if "<image>" in user_text:
            parts = user_text.split("<image>")
            for idx, part in enumerate(parts):
                if part:
                    content.append({"type": "text", "text": part})
                if idx < len(parts) - 1:
                    if not remaining_images:
                        raise ValueError("more <image> placeholders than provided images")
                    content.append(api_image_content(remaining_images.pop(0), quality=self.api_image_quality))
        else:
            for image in remaining_images:
                content.append(api_image_content(image, quality=self.api_image_quality))
            remaining_images = []
            content.append({"type": "text", "text": user_text})

        for image in remaining_images:
            content.append(api_image_content(image, quality=self.api_image_quality))
        messages.append({"role": "user", "content": content})
        return messages

    def _prepare_model_inputs(
        self,
        *,
        system: str,
        user: str,
        images: list[Image.Image],
        call_type: str | None,
        add_generation_prompt: bool,
    ) -> dict[str, Any]:
        if self.processor is None:
            raise RuntimeError("VLM processor is not loaded")

        user_text = user
        messages: list[dict[str, Any]] = []
        if self.prompt_mode == "system_in_user":
            if not (
                call_type == "analyse"
                and not system.strip()
                and user_text.lstrip().startswith("Agent call type: analyse")
            ):
                user_text = build_agent_human_turn(system, user, call_type=call_type)
        elif system:
            messages.append({"role": "system", "content": system})

        if images:
            content = build_interleaved_content(user_text, images)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_text})

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
        }
        if images:
            image_inputs, video_inputs = process_qwen_vision_inputs(messages, images)
            processor_kwargs["images"] = image_inputs
            if video_inputs is not None:
                processor_kwargs["videos"] = video_inputs
        inputs = self.processor(**processor_kwargs)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
            return dict(inputs)
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _prepare_model_inputs_from_messages(
        self,
        *,
        messages: list[dict[str, str]],
        images: list[Image.Image],
        add_generation_prompt: bool,
    ) -> dict[str, Any]:
        if self.processor is None:
            raise RuntimeError("VLM processor is not loaded")
        qwen_messages = build_qwen_messages_from_chat(messages, images)
        text = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        processor_kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if images:
            image_inputs, video_inputs = process_qwen_vision_inputs(qwen_messages, images)
            processor_kwargs["images"] = image_inputs
            if video_inputs is not None:
                processor_kwargs["videos"] = video_inputs
        inputs = self.processor(**processor_kwargs)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
            return dict(inputs)
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    @staticmethod
    def _load_retained_images(memory: Memory) -> list[Image.Image]:
        images: list[Image.Image] = []
        for path in memory.retained_image_paths():
            if not Path(path).exists():
                continue
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        return images


def append_suffix_to_last_user_message(
    messages: list[dict[str, str]],
    suffix: str,
) -> list[dict[str, str]]:
    out = [dict(message) for message in messages]
    for message in reversed(out):
        if message.get("role") == "user":
            message["content"] = str(message.get("content") or "") + suffix
            return out
    raise ValueError("cannot append retry suffix: no user message found")


def build_qwen_messages_from_chat(
    messages: list[dict[str, str]],
    images: list[Image.Image],
) -> list[dict[str, Any]]:
    image_iter = iter(images)
    consumed = 0
    out: list[dict[str, Any]] = []
    for message in messages:
        role = str(message["role"])
        text = str(message.get("content") or "")
        if "<image>" not in text:
            out.append({"role": role, "content": text})
            continue
        content: list[dict[str, Any]] = []
        parts = text.split("<image>")
        for idx, part in enumerate(parts):
            if part:
                content.append({"type": "text", "text": part})
            if idx < len(parts) - 1:
                try:
                    image = next(image_iter)
                except StopIteration as exc:
                    raise ValueError("more <image> placeholders than provided images") from exc
                consumed += 1
                content.append({"type": "image", "image": image})
        out.append({"role": role, "content": content})
    if consumed != len(images):
        raise ValueError(
            f"image placeholder mismatch: consumed {consumed} images for {len(images)} provided images"
        )
    return out


def build_api_messages_from_chat(
    messages: list[dict[str, str]],
    images: list[Image.Image],
    *,
    quality: int,
) -> list[dict[str, Any]]:
    image_iter = iter(images)
    consumed = 0
    out: list[dict[str, Any]] = []
    for message in messages:
        role = str(message["role"])
        text = str(message.get("content") or "")
        if "<image>" not in text:
            out.append({"role": role, "content": text})
            continue
        content: list[dict[str, Any]] = []
        parts = text.split("<image>")
        for idx, part in enumerate(parts):
            if part:
                content.append({"type": "text", "text": part})
            if idx < len(parts) - 1:
                try:
                    image = next(image_iter)
                except StopIteration as exc:
                    raise ValueError("more <image> placeholders than provided images") from exc
                consumed += 1
                content.append(api_image_content(image, quality=quality))
        out.append({"role": role, "content": content})
    if consumed != len(images):
        raise ValueError(
            f"image placeholder mismatch: consumed {consumed} images for {len(images)} provided images"
        )
    return out


def build_interleaved_content(user_text: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    if "<image>" not in user_text:
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image} for image in images
        ]
        content.append({"type": "text", "text": user_text})
        return content

    content = []
    remaining_images = list(images)
    parts = user_text.split("<image>")
    for idx, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if idx < len(parts) - 1:
            if not remaining_images:
                raise ValueError("more <image> placeholders than provided images")
            content.append({"type": "image", "image": remaining_images.pop(0)})
    for image in remaining_images:
        content.append({"type": "image", "image": image})
    return content


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object found in model output: {text[:200]}")
    return json.loads(stripped[start : end + 1])


def extract_tag(text: str, tag: str, *, required: bool) -> str | None:
    match = re.search(
        rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        if required:
            raise ValueError(f"missing <{tag}>...</{tag}> tag")
        return None
    return match.group(1).strip()


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def parse_box_value(value: Any) -> list[float]:
    if isinstance(value, list):
        box = [float(item) for item in value]
        if len(box) != 4:
            raise ValueError(f"expected 4 bbox values, got {box}")
        return box
    if value is None:
        raise ValueError("bbox JSON action requires box")
    return parse_float_list(str(value), expected_len=4)


def split_action_and_args(action_text: str) -> tuple[str, str]:
    stripped = action_text.strip()
    compact = " ".join(stripped.split())
    upper = compact.upper()
    for action in ("UPDATE_GRAPH", "ANSWER", "SEARCH", "CROP"):
        if upper == action:
            return action, ""
        if upper.startswith(action):
            rest = compact[len(action) :].strip()
            rest = rest.lstrip(":- ")
            if rest.startswith("(") and rest.endswith(")"):
                rest = rest[1:-1].strip()
            return action, rest
    raise ValueError(f"unknown action tag content: {action_text!r}")


def extract_named_arg(text: str, name: str) -> str | None:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*(.+?)(?=;\s*\w+\s*=|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    return match.group(1).strip().strip(";")


def strip_field_value(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^(query|search_query)\s*=\s*", "", value, flags=re.IGNORECASE)
    return value.strip().strip("\"'")


def parse_float_list(text: str, *, expected_len: int) -> list[float]:
    values = [
        float(item)
        for item in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(text))
    ]
    if len(values) != expected_len:
        raise ValueError(f"expected {expected_len} float values, got {values}")
    return values


def parse_cells(text: str) -> list[str]:
    cleaned = str(text).strip().strip("[]()").replace('"', "").replace("'", "")
    cells = [cell.strip() for cell in re.split(r"[,\s]+", cleaned) if cell.strip()]
    if not cells:
        raise ValueError("cells argument is empty")
    return cells


def process_qwen_vision_inputs(
    messages: list[dict[str, Any]],
    fallback_images: list[Image.Image],
) -> tuple[list[Image.Image], Any | None]:
    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        return fallback_images, None

    image_inputs, video_inputs = process_vision_info(messages)
    return image_inputs or fallback_images, video_inputs


def api_image_content(image: Image.Image, *, quality: int) -> dict[str, Any]:
    data_url = image_to_data_url(image, quality=quality)
    return {"type": "image_url", "image_url": {"url": data_url}}


def image_to_data_url(image: Image.Image, *, quality: int) -> str:
    buffer = io.BytesIO()
    # PNG = lossless. JPEG (even high q) blurs fine slide text/numbers -> misreads -> lower acc on vLLM path.
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def chat_completion_text(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    max_retries: int,
) -> str:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            m = response_payload["choices"][0]["message"]
            c = str(m.get("content", "") or "")
            r = m.get("reasoning_content")
            if r:
                c = "<think>" + str(r) + "</think>" + c
            elif "</think>" in c and not c.lstrip().startswith("<think>"):
                c = "<think>" + c
            return c
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"API HTTP {exc.code}: {body[:1000]}")
        except Exception as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"API chat completion failed: {last_error}")


def _extract_final_answer(text: str) -> str:
    import re
    t = str(text or "")
    t = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()
    matches = list(re.finditer(r"FINAL\s*ANSWER\s*:\s*(.+)", t, flags=re.IGNORECASE | re.DOTALL))
    if matches:
        ans = matches[-1].group(1).strip()
        return ans.splitlines()[0].strip() if ans else ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines[-1] if lines else t.strip()
