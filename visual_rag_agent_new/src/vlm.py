from __future__ import annotations

"""Minimal VLM wrapper with one entrypoint: generate(messages, images) -> raw text.

Messages are chat dicts ({"role", "content"}); a content string may contain <image> markers,
consumed in order from `images`. Two providers: "qwen" (local transformers model) and
"openai_compatible" (vLLM serve / any chat-completions endpoint; images sent as PNG data URLs).
"""

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from PIL import Image


class VLM:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        adapter_path: str | None = None,
        load_model: bool = True,
        provider: str = "qwen",
        api_base_url: str | None = None,
        api_key_env: str = "MIMO_API_KEY",
        api_timeout: float = 180.0,
        api_max_retries: int = 2,
        api_extra_body: dict[str, Any] | None = None,
    ) -> None:
        if provider not in {"qwen", "openai_compatible"}:
            raise ValueError(f"unsupported provider {provider!r} (qwen | openai_compatible)")
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
        self.api_extra_body = dict(api_extra_body or {})
        self.processor: Any | None = None
        self.model: Any | None = None
        if not self.uses_api and load_model:
            self._load_model()

    @property
    def uses_api(self) -> bool:
        return self.provider == "openai_compatible"

    def generate(self, messages: list[dict[str, str]], images: list[Image.Image], *, max_new_tokens: int | None = None) -> str:
        if self.uses_api:
            return self._generate_api(messages, images, max_new_tokens)
        return self._generate_local(messages, images, max_new_tokens)

    # ------------------------------------------------------------------ local

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        kwargs: dict[str, Any] = {"torch_dtype": getattr(torch, self.dtype, torch.bfloat16), "trust_remote_code": True}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, **kwargs)
        if self.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def _generate_local(self, messages, images, max_new_tokens) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("VLM model is not loaded (load_model=False or API provider)")
        import torch

        qwen_messages = build_qwen_messages_from_chat(messages, images)
        text = self.processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
        kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if images:
            image_inputs, video_inputs = process_qwen_vision_inputs(qwen_messages, images)
            kwargs["images"] = image_inputs
            if video_inputs is not None:
                kwargs["videos"] = video_inputs
        inputs = self.processor(**kwargs).to(self.device)
        gen: dict[str, Any] = {"max_new_tokens": max_new_tokens or self.max_tokens, "do_sample": self.temperature > 0}
        if self.temperature > 0:
            gen["temperature"] = self.temperature
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen)
        prompt_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]

    # ------------------------------------------------------------------ api

    def _generate_api(self, messages, images, max_new_tokens) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is missing for the API VLM provider")
        base_url = (self.api_base_url or os.environ.get("MIMO_BASE_URL") or "").rstrip("/")
        if not base_url:
            raise RuntimeError("api_base_url (or MIMO_BASE_URL) is required for the API VLM provider")
        payload: dict[str, Any] = {
            "model": self.model_path,
            "messages": build_api_messages_from_chat(messages, images),
            "temperature": self.temperature,
            "max_tokens": max_new_tokens or self.max_tokens,
        }
        payload.update(self.api_extra_body)
        return chat_completion_text(base_url=base_url, api_key=api_key, payload=payload,
                                    timeout=self.api_timeout, max_retries=self.api_max_retries)


# ---------------------------------------------------------------------- helpers


def _interleave(messages: list[dict[str, str]], images: list[Image.Image], image_part) -> list[dict[str, Any]]:
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
                    raise ValueError("more <image> markers than images") from exc
                consumed += 1
                content.append(image_part(image))
        out.append({"role": role, "content": content})
    if consumed != len(images):
        raise ValueError(f"image marker mismatch: {consumed} markers for {len(images)} images")
    return out


def build_qwen_messages_from_chat(messages, images) -> list[dict[str, Any]]:
    return _interleave(messages, images, lambda im: {"type": "image", "image": im})


def build_api_messages_from_chat(messages, images) -> list[dict[str, Any]]:
    return _interleave(messages, images, api_image_content)


def process_qwen_vision_inputs(messages: list[dict[str, Any]], fallback_images: list[Image.Image]):
    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        return fallback_images, None
    image_inputs, video_inputs = process_vision_info(messages)
    return image_inputs or fallback_images, video_inputs


def api_image_content(image: Image.Image) -> dict[str, Any]:
    # PNG = lossless: JPEG blurs fine slide text/numbers and causes misreads.
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def chat_completion_text(*, base_url: str, api_key: str, payload: dict[str, Any], timeout: float, max_retries: int) -> str:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body["choices"][0]["message"]
            content = str(message.get("content") or "")
            reasoning = message.get("reasoning_content")
            # Thinking models return their reasoning separately: fold it back into <think> so
            # the protocol parser sees one canonical response.
            if reasoning:
                content = "<think>" + str(reasoning) + "</think>" + content
            elif "</think>" in content and not content.lstrip().startswith("<think>"):
                content = "<think>" + content
            return content
        except urllib.error.HTTPError as exc:
            last_error = RuntimeError(f"API HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:1000]}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        if attempt < max_retries:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"API chat completion failed: {last_error}")
