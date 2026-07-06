from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments

try:
    from transformers import AutoModelForImageTextToText as QwenVLModel
except ImportError:  # pragma: no cover - depends on transformers version
    from transformers import AutoModelForVision2Seq as QwenVLModel

try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # pragma: no cover - text-only smoke still works
    process_vision_info = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for active-graph Qwen3-VL data.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root", default="/root/autodl-tmp/visual_rag_agent")
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--report-to", default="tensorboard")
    return parser.parse_args()


class JsonlDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"SFT data not found: {self.path}")
        self.rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.path}:{line_no}: {exc}") from exc
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No rows in SFT data: {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = dict(self.rows[idx])
        row["_row_index"] = idx
        return row


def normalize_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}

    def convert(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in items:
            role = role_map.get(str(item.get("from") or item.get("role") or "").lower())
            if not role:
                continue
            out.append({"role": role, "content": item.get("value") or item.get("content") or ""})
        return out

    if isinstance(row.get("messages"), list):
        messages = convert(row["messages"])
    elif isinstance(row.get("conversations"), list):
        messages = convert(row["conversations"])
    elif "prompt" in row and ("response" in row or "target" in row or "completion" in row):
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row.get("response") or row.get("target") or row.get("completion") or ""},
        ]
    elif "instruction" in row and ("output" in row or "response" in row):
        prompt = str(row["instruction"])
        if row.get("input"):
            prompt += "\n" + str(row["input"])
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": row.get("output") or row.get("response") or ""}]
    else:
        keys = ", ".join(sorted(k for k in row.keys() if not k.startswith("_")))
        raise ValueError(
            "SFT row has no assistant target. Expected messages/conversations or prompt+response. "
            f"row_index={row.get('_row_index')} keys=[{keys}]"
        )

    target = str(row.get("target") or "").strip()
    if target:
        while messages and messages[-1].get("role") == "assistant" and str(messages[-1].get("content") or "").strip() != target:
            messages.pop()
        if not messages or messages[-1].get("role") != "assistant" or str(messages[-1].get("content") or "").strip() != target:
            messages.append({"role": "assistant", "content": target})

    if not any(msg.get("role") == "assistant" for msg in messages):
        raise ValueError(f"SFT row has no assistant message. row_index={row.get('_row_index')}")
    return [dict(msg) for msg in messages]


def resolve_media_paths(value: Any, image_root: Path) -> Any:
    if isinstance(value, list):
        return [resolve_media_paths(item, image_root) for item in value]
    if isinstance(value, dict):
        item = dict(value)
        if item.get("type") in {"image", "image_url"}:
            for key in ("image", "path", "url"):
                if key in item and isinstance(item[key], str):
                    item[key] = resolve_one_path(item[key], image_root)
        return {key: resolve_media_paths(val, image_root) if key != "image" else val for key, val in item.items()}
    return value


def resolve_one_path(path_text: str, image_root: Path) -> str:
    if path_text.startswith(("http://", "https://", "file://", "data:")):
        return path_text
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    candidates = [
        image_root / path,
        image_root / "data" / path,
        image_root / "data" / "images" / path,
        image_root / "data" / "slidevqa" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(image_root / path)


def last_assistant_index(messages: list[dict[str, Any]]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            return idx
    raise ValueError("No assistant message")


def collect_vision(messages: list[dict[str, Any]]) -> tuple[list[Any] | None, list[Any] | None]:
    if process_vision_info is None:
        return None, None
    images, videos = process_vision_info(messages)
    return images or None, videos or None


class DataCollator:
    def __init__(self, processor: Any, image_root: str | Path, max_length: int):
        self.processor = processor
        self.image_root = Path(image_root)
        self.max_length = max_length
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_texts: list[str] = []
        full_images: list[Any] = []
        full_videos: list[Any] = []
        prompt_lengths: list[int] = []

        for row in rows:
            messages = normalize_messages(row)
            for msg in messages:
                msg["content"] = resolve_media_paths(msg.get("content", ""), self.image_root)
            assistant_idx = last_assistant_index(messages)
            prompt_messages = messages[:assistant_idx]

            full_texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

            images, videos = collect_vision(messages)
            if images:
                full_images.extend(images)
            if videos:
                full_videos.extend(videos)

            prompt_images, prompt_videos = collect_vision(prompt_messages)
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=prompt_images,
                videos=prompt_videos,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_lengths.append(int(prompt_inputs["attention_mask"][0].sum().item()))

        batch = self.processor(
            text=full_texts,
            images=full_images or None,
            videos=full_videos or None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, : min(prompt_len, labels.shape[1])] = -100
        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()
    train_dataset = JsonlDataset(args.train_data)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = QwenVLModel.from_pretrained(args.model_path, **model_kwargs)
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to=[item for item in args.report_to.split(",") if item],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(processor=processor, image_root=args.image_root, max_length=args.max_length),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
