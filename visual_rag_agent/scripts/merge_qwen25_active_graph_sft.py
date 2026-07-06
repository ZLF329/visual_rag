#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description='Merge Qwen2.5-VL Active Graph SFT LoRA adapter into a full model.')
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--attn-implementation', default='flash_attention_2')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    base = Path(args.base_model)
    adapter = Path(args.adapter)
    output = Path(args.output)
    if not base.exists():
        raise FileNotFoundError(f'base model not found: {base}')
    if not adapter.exists():
        raise FileNotFoundError(f'adapter not found: {adapter}')
    if output.exists() and not args.force:
        if (output / 'config.json').exists():
            print(f'[merge] output already exists, skipping: {output}')
            return
    tmp = output.with_name(output.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    if output.exists():
        shutil.rmtree(output)

    print(f'[merge] base={base}', flush=True)
    print(f'[merge] adapter={adapter}', flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(base),
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation or None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(str(base), trust_remote_code=True)
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(tmp), safe_serialization=True)
    processor.save_pretrained(str(tmp))
    tmp.rename(output)
    print(f'[merge] wrote {output}', flush=True)


if __name__ == '__main__':
    main()
