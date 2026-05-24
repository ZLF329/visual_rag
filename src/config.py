from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "models": {
        "vlm": {
            "provider": "qwen",
            "name": "Qwen/Qwen3-VL-Instruct-4B",
            "max_tokens": 1024,
            "temperature": 0.0,
        },
        "retriever": {
            "name": "/root/autodl-tmp/models/Qwen3-VL-Embedding-8B",
            "index_path": "data/indexes/slidevqa",
        },
    },
    "agent": {"top_k": 1, "max_iters": 5, "partial_memory_capacity": 2},
    "image_budget": {"yes_pixels": 400_000, "partial_pixels": 200_000},
    "dataset": {"name": "slidevqa", "split": "test", "num_samples": 200},
    "runtime": {
        "output_dir": "outputs/runs",
        "device": "cuda",
        "dtype": "bfloat16",
        "attn_implementation": None,
    },
}


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return deep_merge(config, loaded)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
