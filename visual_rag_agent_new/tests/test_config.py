import tempfile
from pathlib import Path

from src.config import load_config

ROOT = Path(__file__).resolve().parent.parent


def test_config_overlays_merge_in_order():
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.yaml", Path(tmp) / "b.yaml"
        a.write_text("agent:\n  max_turns: 12\nruntime:\n  attn_implementation: x\n", encoding="utf-8")
        b.write_text("runtime:\n  attn_implementation: y\n", encoding="utf-8")
        config = load_config([a, b])
        assert config["agent"]["max_turns"] == 12 and config["runtime"]["attn_implementation"] == "y"
        assert config["agent"]["frame"] == "qwen2.5-vl"                     # untouched defaults survive
        assert load_config(a)["runtime"]["attn_implementation"] == "x"


def test_blackwell_overlay_only_swaps_the_attention_kernel():
    base = load_config(ROOT / "config" / "teacher.yaml")
    both = load_config([ROOT / "config" / "teacher.yaml", ROOT / "config" / "blackwell.yaml"])
    assert both["runtime"]["attn_implementation"] == "flash_attention_2"
    both["runtime"]["attn_implementation"] = base["runtime"]["attn_implementation"]
    assert both == base
