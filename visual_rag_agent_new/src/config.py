from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "models": {
        "vlm": {
            "provider": "openai_compatible",
            "name": "new",
            "api_base_url": "http://localhost:8000/v1",
            "api_key_env": "OPENAI_API_KEY",
            "max_tokens": 1024,
            "temperature": 0.0,
        },
        "retriever": {
            "name": "/root/autodl-tmp/models/Qwen3-VL-Embedding-8B",
            "index_path": "data/indexes/slidevqa",
        },
    },
    "agent": {"max_turns": 10, "retrieve_k": 5, "frame": "qwen2.5-vl", "retrieval_scope": "flat",
              "first_search": "model", "search_guidance": False, "image_max_pixels": None},
    "runtime": {"device": "cuda", "dtype": "bfloat16", "attn_implementation": None},
}


def load_config(paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None) -> dict[str, Any]:
    """Defaults, then each YAML file in order; later files override earlier ones (overlays such as
    config/blackwell.yaml change only what differs on one machine)."""
    config = deepcopy(DEFAULT_CONFIG)
    if paths is None:
        return config
    for path in [paths] if isinstance(paths, (str, Path)) else paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            config = deep_merge(config, yaml.safe_load(f) or {})
    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def build_vlm(config: dict[str, Any], *, device: str | None = None):
    from src.vlm import VLM

    vlm_cfg = config["models"]["vlm"]
    runtime = config["runtime"]
    return VLM(
        model_path=vlm_cfg["name"],
        device=device or runtime["device"],
        max_tokens=int(vlm_cfg.get("max_tokens", 1024)),
        temperature=float(vlm_cfg.get("temperature", 0.0)),
        dtype=runtime["dtype"],
        attn_implementation=runtime.get("attn_implementation"),
        adapter_path=vlm_cfg.get("adapter_path"),
        provider=vlm_cfg.get("provider", "openai_compatible"),
        api_base_url=vlm_cfg.get("api_base_url"),
        api_key_env=vlm_cfg.get("api_key_env", "OPENAI_API_KEY"),
        api_timeout=float(vlm_cfg.get("api_timeout", 180.0)),
        api_max_retries=int(vlm_cfg.get("api_max_retries", 2)),
        api_extra_body=vlm_cfg.get("api_extra_body") or {},
    )


def build_retriever(config: dict[str, Any], *, index_path: str | None = None, device: str | None = None):
    from src.retriever import make_retriever

    cfg = config["models"]["retriever"]
    return make_retriever(
        cfg["name"],
        index_path or cfg["index_path"],
        device=device or config["runtime"]["device"],
        dtype=config["runtime"]["dtype"],
        attn_implementation=config["runtime"].get("attn_implementation"),
        bank_device=cfg.get("bank_device"),
    )
