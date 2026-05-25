#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen3-VL-4B-Instruct"
DEFAULT_TRAIN_FILE = (
    "outputs/sft_trajectories/gpt5mini_20260525_004442/kept_sft_trajectories.jsonl"
)
DEFAULT_IMAGE_ROOT = "data/corpora/slidevqa_sft/pages"
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/sft_qwen3vl4b_visor"


@dataclass
class SftExample:
    system: str
    user: str
    target: str
    image_paths: list[str]
    sample_id: str
    step: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Completion-only SFT for Qwen-VL on agent traces.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--image-root", action="append", default=[DEFAULT_IMAGE_ROOT])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument("--freeze-vision", action="store_true", default=True)
    parser.add_argument("--no-freeze-vision", dest="freeze_vision", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoProcessor, Trainer, TrainingArguments

    model = load_model(args)
    if args.freeze_vision:
        freeze_visual_parameters(model)
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = TrajectorySftDataset(
        train_file=Path(args.train_file),
        image_roots=[Path(p) for p in args.image_root],
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
        skip_missing_images=args.skip_missing_images,
    )
    print(json.dumps(dataset.summary(), ensure_ascii=False, indent=2))

    collator = QwenVlSftCollator(processor=processor, max_length=args.max_length)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=[] if args.report_to == "none" else args.report_to.split(","),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


def load_model(args: argparse.Namespace) -> Any:
    from transformers import AutoModelForVision2Seq

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    errors: list[str] = []
    for cls in [AutoModelForImageTextToText, AutoModelForVision2Seq]:
        if cls is None:
            continue
        try:
            return cls.from_pretrained(args.model_path, **kwargs)
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError("failed to load VLM:\n" + "\n".join(errors))


def freeze_visual_parameters(model: torch.nn.Module) -> None:
    frozen = 0
    trainable = 0
    visual_markers = (
        "visual",
        "vision_tower",
        "vision_model",
        "multi_modal_projector",
        "mm_projector",
        "merger",
    )
    for name, param in model.named_parameters():
        if any(marker in name for marker in visual_markers):
            param.requires_grad = False
            frozen += param.numel()
        else:
            trainable += param.numel()
    print(
        json.dumps(
            {
                "frozen_visual_parameters": frozen,
                "trainable_parameters": trainable,
            },
            ensure_ascii=False,
        )
    )


class TrajectorySftDataset(Dataset[SftExample]):
    def __init__(
        self,
        *,
        train_file: Path,
        image_roots: list[Path],
        limit: int | None,
        shuffle: bool,
        seed: int,
        skip_missing_images: bool,
    ) -> None:
        self.resolver = ImageResolver(image_roots)
        self.examples: list[SftExample] = []
        self.skipped_missing_images = 0
        self.step_counts: dict[str, int] = {}
        self._load(
            train_file=train_file,
            limit=limit,
            skip_missing_images=skip_missing_images,
        )
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SftExample:
        return self.examples[idx]

    def summary(self) -> dict[str, Any]:
        return {
            "sft_examples": len(self.examples),
            "step_counts": self.step_counts,
            "skipped_missing_images": self.skipped_missing_images,
        }

    def _load(
        self,
        *,
        train_file: Path,
        limit: int | None,
        skip_missing_images: bool,
    ) -> None:
        with train_file.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if limit is not None and line_idx >= limit:
                    break
                if not line.strip():
                    continue
                self._add_row(json.loads(line), skip_missing_images=skip_missing_images)

    def _add_row(self, row: dict[str, Any], *, skip_missing_images: bool) -> None:
        sample_id = str(row.get("sample_id", row.get("row_index", "unknown")))
        question = row.get("question") or row.get("query") or ""
        compact_summary = ""
        compact_key_facts: list[str] = []
        confirmed_images: list[str] = []
        partial_images: deque[str] = deque(maxlen=2)
        current_pages: dict[str, str] = {}
        latest_analysis: dict[str, Any] | None = None

        for item in row.get("trace", []):
            step = item.get("step")
            if step == "search":
                current_pages = self._resolve_search_pages(
                    item.get("pages", []) or [],
                    skip_missing_images=skip_missing_images,
                )
                continue

            if step == "decide":
                system, user = build_decide_prompt_safe(
                    original_query=question,
                    memory_context=context_for_decide(compact_summary, compact_key_facts),
                )
                image_paths = [*confirmed_images, *partial_images]
                self._append_example(
                    system=system,
                    user=user,
                    target=item.get("raw_text") or dump_json(item.get("result", {})),
                    image_paths=image_paths,
                    sample_id=sample_id,
                    step="decide",
                )
                continue

            if step == "analyse":
                page_label = str(item.get("page") or "")
                image_path = current_pages.get(page_label)
                if not image_path:
                    parsed = parse_page_label(page_label)
                    if parsed is not None:
                        image_path = self.resolver.resolve(parsed[0], parsed[1])
                if not image_path:
                    self.skipped_missing_images += 1
                    if skip_missing_images:
                        continue
                    raise FileNotFoundError(
                        f"missing image for sample={sample_id} page={page_label}. "
                        "Run scripts/materialize_sft_images.py first or pass --skip-missing-images."
                    )
                system, user = build_analyse_prompt_safe(
                    original_query=question,
                    memory_context=context_for_analyse(compact_summary, compact_key_facts),
                )
                self._append_example(
                    system=system,
                    user=user,
                    target=item.get("raw_text") or dump_json(item.get("result", {})),
                    image_paths=[image_path],
                    sample_id=sample_id,
                    step="analyse",
                )
                latest_analysis = item.get("result") or {}
                compact_summary = append_summary(
                    compact_summary,
                    str(latest_analysis.get("summary") or ""),
                    search_query=item.get("query"),
                )
                judge = latest_analysis.get("judge") or latest_analysis.get("decision")
                if judge == "yes":
                    confirmed_images.append(image_path)
                elif judge == "partial":
                    partial_images.append(image_path)
                continue

            if step == "memory_update":
                if latest_analysis is None:
                    continue
                system, user = build_memory_update_prompt_safe(
                    original_query=question,
                    previous_key_facts=compact_key_facts,
                    latest_judge=str(latest_analysis.get("judge") or latest_analysis.get("decision") or ""),
                    latest_key_facts=latest_analysis.get("key_facts") or [],
                )
                self._append_example(
                    system=system,
                    user=user,
                    target=item.get("raw_text") or dump_json(item.get("result", {})),
                    image_paths=[],
                    sample_id=sample_id,
                    step="memory_update",
                )
                compact_key_facts = [
                    str(f).strip()
                    for f in (item.get("result") or {}).get("key_facts", [])
                    if str(f).strip()
                ]

    def _resolve_search_pages(
        self,
        pages: list[Any],
        *,
        skip_missing_images: bool,
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for page in pages:
            label = ""
            image_path = None
            if isinstance(page, dict):
                label = str(page.get("page_label") or "")
                raw_path = page.get("image_path")
                if raw_path and Path(raw_path).exists():
                    image_path = str(Path(raw_path))
                page_num = page.get("page_num")
                if not image_path and label and page_num:
                    deck = label.split("/page_")[0]
                    image_path = self.resolver.resolve(deck, int(page_num))
            elif isinstance(page, str):
                label = page
            if not image_path and label:
                parsed = parse_page_label(label)
                if parsed is not None:
                    image_path = self.resolver.resolve(parsed[0], parsed[1])
            if image_path:
                resolved[label] = image_path
            elif not skip_missing_images and label:
                raise FileNotFoundError(f"missing image for search page={label}")
        return resolved

    def _append_example(
        self,
        *,
        system: str,
        user: str,
        target: str,
        image_paths: list[str],
        sample_id: str,
        step: str,
    ) -> None:
        if not target.strip():
            return
        self.examples.append(
            SftExample(
                system=system,
                user=user,
                target=target.strip(),
                image_paths=image_paths,
                sample_id=sample_id,
                step=step,
            )
        )
        self.step_counts[step] = self.step_counts.get(step, 0) + 1


class ImageResolver:
    def __init__(self, image_roots: list[Path]) -> None:
        self.image_roots = image_roots
        self.manifests: list[dict[str, str]] = []
        for root in image_roots:
            manifest = root / "manifest.json"
            if manifest.exists():
                self.manifests.append(json.loads(manifest.read_text(encoding="utf-8")))

    def resolve(self, deck: str, page_num: int) -> str | None:
        label = f"{deck}/page_{page_num:02d}"
        for manifest in self.manifests:
            candidate = manifest.get(label)
            if candidate and Path(candidate).exists():
                return str(Path(candidate))
        names = [
            Path(safe_name(deck)) / f"page_{page_num:02d}.jpg",
            Path(safe_name(deck)) / f"page_{page_num:02d}.png",
            Path(deck) / f"page_{page_num:02d}.jpg",
            Path(deck) / f"page_{page_num:02d}.png",
        ]
        for root in self.image_roots:
            for name in names:
                candidate = root / name
                if candidate.exists():
                    return str(candidate)
        return None


class QwenVlSftCollator:
    def __init__(self, *, processor: Any, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length
        tokenizer = processor.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list[SftExample]) -> dict[str, torch.Tensor]:
        encoded = [self._encode_one(feature) for feature in features]
        batch = {
            "input_ids": pad_sequence(
                [item["input_ids"] for item in encoded],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [item["attention_mask"] for item in encoded],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [item["labels"] for item in encoded],
                batch_first=True,
                padding_value=-100,
            ),
        }
        sequence_pad_values = {
            "mm_token_type_ids": 0,
        }
        for key, pad_value in sequence_pad_values.items():
            tensors = [item[key] for item in encoded if key in item]
            if tensors:
                batch[key] = pad_sequence(
                    tensors,
                    batch_first=True,
                    padding_value=pad_value,
                )

        extra_keys = sorted({key for item in encoded for key in item} - set(batch))
        for key in extra_keys:
            tensors = [item[key] for item in encoded if key in item]
            if not tensors:
                continue
            batch[key] = torch.cat(tensors, dim=0)
        return batch

    def _encode_one(self, example: SftExample) -> dict[str, torch.Tensor]:
        images = load_images(example.image_paths)
        prompt_messages = build_messages(example.system, example.user, images)
        full_messages = [
            *prompt_messages,
            {"role": "assistant", "content": example.target},
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        image_inputs, video_inputs = process_vision_inputs(full_messages, images)
        full_inputs = self._processor_call(full_text, image_inputs, video_inputs)
        prompt_inputs = self._processor_call(prompt_text, image_inputs, video_inputs)

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        if self.max_length and input_ids.numel() > self.max_length:
            raise ValueError(
                f"sample={example.sample_id} step={example.step} length={input_ids.numel()} "
                f"exceeds --max-length={self.max_length}; raise max length or filter the sample."
            )
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        out: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key, value in full_inputs.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            if isinstance(value, torch.Tensor):
                if key == "mm_token_type_ids" and value.ndim == 2 and value.shape[0] == 1:
                    out[key] = value[0]
                else:
                    out[key] = value
        return out

    def _processor_call(
        self,
        text: str,
        image_inputs: list[Any] | None,
        video_inputs: Any | None,
    ) -> dict[str, torch.Tensor]:
        kwargs: dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
            "padding": False,
        }
        if image_inputs:
            kwargs["images"] = image_inputs
        if video_inputs is not None:
            kwargs["videos"] = video_inputs
        return self.processor(**kwargs)


def build_messages(system: str, user: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if images:
        content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": user})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": user})
    return messages


def load_images(paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def process_vision_inputs(
    messages: list[dict[str, Any]],
    fallback_images: list[Image.Image],
) -> tuple[list[Any] | None, Any | None]:
    if not fallback_images:
        return None, None
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        return image_inputs or fallback_images, video_inputs
    except Exception:
        return fallback_images, None


def build_decide_prompt_safe(*, original_query: str, memory_context: str) -> tuple[str, str]:
    from src.prompts import build_decide_prompt

    return build_decide_prompt(original_query=original_query, memory_context=memory_context)


def build_analyse_prompt_safe(*, original_query: str, memory_context: str) -> tuple[str, str]:
    from src.prompts import build_analyse_prompt

    return build_analyse_prompt(original_query=original_query, memory_context=memory_context)


def build_memory_update_prompt_safe(
    *,
    original_query: str,
    previous_key_facts: list[str],
    latest_judge: str,
    latest_key_facts: list[str],
) -> tuple[str, str]:
    from src.prompts import build_memory_update_prompt

    return build_memory_update_prompt(
        original_query=original_query,
        previous_key_facts=previous_key_facts,
        latest_judge=latest_judge,
        latest_key_facts=latest_key_facts,
    )


def context_for_decide(summary: str, key_facts: list[str]) -> str:
    return "Compact memory for next step:\n" + compact_memory_text(summary, key_facts)


def context_for_analyse(summary: str, key_facts: list[str]) -> str:
    return "Compact memory for current question:\n" + compact_memory_text(summary, key_facts)


def compact_memory_text(summary: str, key_facts: list[str]) -> str:
    if not summary and not key_facts:
        return "  None yet."
    return f"  summary: {summary or 'None'}\n  key_facts: {key_facts or []}"


def append_summary(current: str, summary: str, *, search_query: str | None) -> str:
    summary = summary.strip()
    if not summary:
        return current
    if search_query:
        summary = f"[query: {search_query}] {summary}"
    return f"{current}\n{summary}" if current else summary


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def parse_page_label(label: str) -> tuple[str, int] | None:
    match = re.match(r"(.+)/page_0*(\d+)$", str(label))
    if not match:
        return None
    return match.group(1), int(match.group(2))


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)[:160]


if __name__ == "__main__":
    main()
