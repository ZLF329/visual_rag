#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LEGACY_BEGIN_MARKER = "# BEGIN VISOR_TRAJECTORY_SFT_DATASET"
LEGACY_END_MARKER = "# END VISOR_TRAJECTORY_SFT_DATASET"
BEGIN_PREFIX = "# BEGIN LOCAL_QWENVL_DATASET"
END_PREFIX = "# END LOCAL_QWENVL_DATASET"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a local dataset in QwenLM/Qwen3-VL qwen-vl-finetune."
    )
    parser.add_argument(
        "--qwen-finetune-root",
        default="/scratch/punim0614/lifuzhang/Qwen3-VL/qwen-vl-finetune",
        help="Path to the official qwen-vl-finetune directory.",
    )
    parser.add_argument("--dataset-name", default="visor_trajectory_sft")
    parser.add_argument("--annotation-path", required=True)
    parser.add_argument("--data-path", default="")
    args = parser.parse_args()

    root = Path(args.qwen_finetune_root)
    init_path = root / "qwenvl" / "data" / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"QwenVL data registry not found: {init_path}")

    text = init_path.read_text(encoding="utf-8")
    text = sanitize_registry_text(text)
    text = remove_existing_dataset_blocks(text, args.dataset_name)

    dataset_var = safe_var_name(args.dataset_name)
    begin_marker = f"{BEGIN_PREFIX} {args.dataset_name}"
    end_marker = f"{END_PREFIX} {args.dataset_name}"
    block = "\n".join(
        [
            "",
            begin_marker,
            f"{dataset_var} = {{",
            f"    \"annotation_path\": {json.dumps(str(Path(args.annotation_path).resolve()))},",
            f"    \"data_path\": {json.dumps(args.data_path)},",
            "}",
            f"data_dict[{json.dumps(args.dataset_name)}] = {dataset_var}",
            end_marker,
            "",
        ]
    )
    init_path.write_text(text.rstrip() + block, encoding="utf-8")
    print(
        json.dumps(
            {
                "registry": str(init_path),
                "dataset_name": args.dataset_name,
                "annotation_path": str(Path(args.annotation_path).resolve()),
                "data_path": args.data_path,
            },
            indent=2,
        )
    )



def sanitize_registry_text(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        if re.fullmatch(r"_eval\d+", line.strip()):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).rstrip() + "\n"

def remove_existing_dataset_blocks(text: str, dataset_name: str) -> str:
    markers = [
        (LEGACY_BEGIN_MARKER, LEGACY_END_MARKER),
        (f"{BEGIN_PREFIX} {dataset_name}", f"{END_PREFIX} {dataset_name}"),
    ]
    target = f"data_dict[{json.dumps(dataset_name)}]"
    for begin_marker, end_marker in markers:
        while True:
            start = text.find(begin_marker)
            if start == -1:
                break
            end = text.find(end_marker, start)
            if end == -1:
                raise ValueError(f"found {begin_marker} without {end_marker}")
            end += len(end_marker)
            block = text[start:end]
            if begin_marker.startswith(BEGIN_PREFIX) or target in block:
                text = text[:start].rstrip() + "\n" + text[end:].lstrip()
            else:
                break
    return text


def safe_var_name(name: str) -> str:
    out = "".join(ch.upper() if ch.isalnum() else "_" for ch in name)
    out = out.strip("_") or "LOCAL_DATASET"
    if out[0].isdigit():
        out = f"DATASET_{out}"
    return out


if __name__ == "__main__":
    main()
