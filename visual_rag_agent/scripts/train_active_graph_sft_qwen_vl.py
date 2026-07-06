#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

from src.vlm import process_qwen_vision_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA SFT for Active-Clue-Graph dynamic-chat trajectories."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated modules. Default follows Qwen-VL attention-only LoRA.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = dtype_from_name(args.dtype)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation or None,
        trust_remote_code=True,
    )
    if args.device:
        model.to(args.device)

    if args.lora_r > 0:
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as exc:  # pragma: no cover - depends on remote env
            raise RuntimeError("peft is required for --lora-r > 0") from exc
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                item.strip()
                for item in args.lora_target_modules.split(",")
                if item.strip()
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_dataset = ActiveGraphSftDataset(Path(args.train_file), limit=args.limit, shuffle=True, seed=args.seed)
    if len(train_dataset) == 0:
        raise ValueError(f"no usable SFT rows found in {args.train_file}")
    eval_dataset = (
        ActiveGraphSftDataset(Path(args.eval_file), limit=args.eval_limit, shuffle=False, seed=args.seed)
        if args.eval_file
        else None
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=lambda rows: list(rows),
    )
    total_update_steps = compute_total_update_steps(
        examples=len(train_dataset),
        epochs=args.epochs,
        batch_size=args.per_device_train_batch_size,
        grad_accum=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("model has no trainable parameters")
    optimizer = AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(0, int(total_update_steps * args.warmup_ratio)),
        num_training_steps=max(1, total_update_steps),
    )

    model.train()
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    batches_per_epoch = max(1, math.ceil(len(train_dataset) / max(1, args.per_device_train_batch_size)))
    target_micro_steps = (
        math.ceil(args.epochs * batches_per_epoch)
        if args.max_steps < 0
        else args.max_steps * args.gradient_accumulation_steps
    )
    progress = tqdm(total=target_micro_steps, desc="sft")
    while micro_step < target_micro_steps:
        for rows in loader:
            if micro_step >= target_micro_steps:
                break
            batch = encode_rows(
                rows,
                processor=processor,
                device=args.device,
                max_length=args.max_length,
            )
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            micro_step += 1
            progress.update(1)

            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / max(1, args.logging_steps * args.gradient_accumulation_steps)
                    print(json.dumps({"step": global_step, "loss": avg_loss}, ensure_ascii=False), flush=True)
                    running_loss = 0.0

                if eval_dataset is not None and global_step % args.eval_steps == 0:
                    eval_loss = evaluate_loss(
                        model,
                        processor,
                        eval_dataset,
                        device=args.device,
                        max_length=args.max_length,
                        batch_size=args.per_device_eval_batch_size,
                    )
                    print(json.dumps({"step": global_step, "eval_loss": eval_loss}, ensure_ascii=False), flush=True)

                if global_step % args.save_steps == 0:
                    save_model(model, processor, output_dir / f"checkpoint-{global_step}")

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    if micro_step % args.gradient_accumulation_steps != 0 and (args.max_steps < 0 or global_step < args.max_steps):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
    progress.close()
    save_model(model, processor, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "global_step": global_step}, ensure_ascii=False, indent=2))


class ActiveGraphSftDataset(Dataset):
    def __init__(self, path: Path, *, limit: int | None, shuffle: bool, seed: int) -> None:
        self.rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("messages") or not row.get("target"):
                    continue
                self.rows.append(row)
                if limit is not None and len(self.rows) >= limit:
                    break
        if shuffle:
            random.Random(seed).shuffle(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def encode_rows(
    rows: list[dict[str, Any]],
    *,
    processor: Any,
    device: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    encoded = [
        encode_row(row, processor=processor, device="", max_length=max_length)
        for row in rows
    ]
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    batch = collate_encoded_rows(encoded, pad_token_id=pad_token_id)
    return {
        key: value.to(device) if device and hasattr(value, "to") else value
        for key, value in batch.items()
    }


def collate_encoded_rows(
    rows: list[dict[str, torch.Tensor]],
    *,
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("empty batch")
    max_length = max(int(row["input_ids"].shape[-1]) for row in rows)
    out: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "labels"):
        if key not in rows[0]:
            continue
        if key == "labels":
            pad_value = -100
        elif key == "input_ids":
            pad_value = pad_token_id
        else:
            pad_value = 0
        padded = []
        for row in rows:
            value = row[key]
            pad_len = max_length - int(value.shape[-1])
            if pad_len > 0:
                pad = value.new_full((value.shape[0], pad_len), pad_value)
                value = torch.cat([pad, value], dim=-1)
            padded.append(value)
        out[key] = torch.cat(padded, dim=0)
    extra_keys = sorted(set().union(*(row.keys() for row in rows)) - set(out.keys()))
    for key in extra_keys:
        values = [row[key] for row in rows if key in row and row[key] is not None]
        if not values:
            continue
        if all(torch.is_tensor(value) for value in values):
            out[key] = torch.cat(values, dim=0)
        else:
            out[key] = values
    return out


def encode_row(
    row: dict[str, Any],
    *,
    processor: Any,
    device: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    image_paths = [str(path) for path in row.get("image_paths") or []]
    images = load_images(image_paths)
    prompt_messages = build_qwen_messages(row["messages"], images)
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": str(row["target"])},
    ]
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    image_inputs, video_inputs = process_qwen_vision_inputs(full_messages, images)
    full_inputs = processor_call(processor, full_text, image_inputs, video_inputs)
    prompt_inputs = processor_call(processor, prompt_text, image_inputs, video_inputs)
    input_ids = full_inputs["input_ids"]
    if max_length and input_ids.shape[-1] > max_length:
        raise ValueError(
            f"sample={row.get('sample_id')} id={row.get('id')} length={input_ids.shape[-1]} "
            f"exceeds --max-length={max_length}"
        )
    labels = input_ids.clone()
    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    labels[:, :prompt_len] = -100
    if "attention_mask" in full_inputs:
        labels[full_inputs["attention_mask"] == 0] = -100
    full_inputs["labels"] = labels
    return {
        key: value.to(device) if device and hasattr(value, "to") else value
        for key, value in full_inputs.items()
    }


def build_qwen_messages(messages: list[dict[str, Any]], images: list[Image.Image]) -> list[dict[str, Any]]:
    image_iter = iter(images)
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
                    raise ValueError("more <image> placeholders than image_paths") from exc
                content.append({"type": "image", "image": image})
        out.append({"role": role, "content": content})
    return out


def processor_call(processor: Any, text: str, image_inputs: Any, video_inputs: Any) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt", "padding": False}
    if image_inputs:
        kwargs["images"] = image_inputs
    if video_inputs is not None:
        kwargs["videos"] = video_inputs
    return dict(processor(**kwargs))


@torch.no_grad()
def evaluate_loss(
    model: Any,
    processor: Any,
    dataset: ActiveGraphSftDataset,
    *,
    device: str,
    max_length: int,
    batch_size: int,
) -> float:
    model.eval()
    losses: list[float] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda rows: list(rows))
    for rows in loader:
        batch = encode_rows(rows, processor=processor, device=device, max_length=max_length)
        losses.append(float(model(**batch).loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_model(model: Any, processor: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def load_images(paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def compute_total_update_steps(
    *,
    examples: int,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    max_steps: int,
) -> int:
    if max_steps > 0:
        return max_steps
    micro_batches = math.ceil(examples * epochs / max(1, batch_size))
    return max(1, math.ceil(micro_batches / max(1, grad_accum)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
