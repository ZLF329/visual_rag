from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image
from pydantic import BaseModel

from src.memory import Memory
from src.prompts import (
    STRICT_JSON_RETRY_SUFFIX,
    build_analyse_prompt,
    build_decide_prompt,
    build_memory_update_prompt,
)
from src.baseline_prompts import (
    build_image_summary_prompt,
    build_summary_decide_prompt,
)
from src.schemas import AnalyseResult, DecideResult, ImageSummaryResult, MemoryUpdateResult


T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    def __init__(self, schema_name: str, outputs: list[str], last_error: Exception) -> None:
        super().__init__(
            f"failed to parse {schema_name} after {len(outputs)} attempt(s): {last_error}"
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
        load_model: bool = True,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.processor: Any | None = None
        self.model: Any | None = None
        if load_model:
            self._load_model()

    def decide_next(
        self,
        original_query: str,
        memory: Memory,
        warnings_summary: str | None = None,
    ) -> DecideResult:
        memory_context = warnings_summary or memory.context_for_decide()
        system, user = build_decide_prompt(
            original_query=original_query,
            memory_context=memory_context,
        )
        images = self._load_retained_images(memory)
        return self._generate_structured(
            system=system,
            user=user,
            images=images,
            schema=DecideResult,
            validator=lambda item: item.validate_branch(),
        )

    def analyse(
        self,
        image: Image.Image,
        original_query: str,
        memory_context: str,
    ) -> AnalyseResult:
        system, user = build_analyse_prompt(
            original_query=original_query,
            memory_context=memory_context,
        )
        return self._generate_structured(
            system=system,
            user=user,
            images=[image],
            schema=AnalyseResult,
            validator=lambda item: item.validate_branch(),
        )


    def update_memory(
        self,
        *,
        original_query: str,
        memory: Memory,
        latest_analysis: AnalyseResult,
        max_new_tokens: int = 1000,
    ) -> MemoryUpdateResult:
        system, user = build_memory_update_prompt(
            original_query=original_query,
            previous_key_facts=memory.consolidated_key_facts,
            latest_judge=latest_analysis.decision,
            latest_key_facts=[] if latest_analysis.judge == "no" else latest_analysis.key_facts,
        )
        return self._generate_structured(
            system=system,
            user=user,
            images=[],
            schema=MemoryUpdateResult,
            validator=lambda item: item.validate_branch(),
            max_new_tokens=max_new_tokens,
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
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "VLM model is not loaded. Instantiate with load_model=True on a GPU machine."
            )

        import torch

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        if images:
            content: list[dict[str, Any]] = [
                {"type": "image", "image": image} for image in images
            ]
            content.append({"type": "text", "text": user})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user})

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
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
        else:
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

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

    @staticmethod
    def _load_retained_images(memory: Memory) -> list[Image.Image]:
        images: list[Image.Image] = []
        for path in memory.retained_image_paths():
            if not Path(path).exists():
                continue
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        return images


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
