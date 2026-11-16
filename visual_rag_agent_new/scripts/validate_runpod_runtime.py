#!/usr/bin/env python3
"""Validate migrated data, the actual verl env import, and one local GPU retrieval.

No policy training or external answer-judge requests are made.
"""
import json
import os
from pathlib import Path
import time

import pyarrow.parquet as pq
import torch
import yaml

from agent_system.environments.env_package.slidevqa.envs import SlideVQAMultiProcessEnv
from src.retriever import ColQwenRetriever


def main():
    root = Path(os.environ.get("WORK_ROOT", "/workspace"))
    vra = root / "code/visual_rag_agent_new"
    config = {}
    for line in (vra / "rl/experiments/hybrid_coverage_gspo.args").read_text().splitlines():
        key, value = line.lstrip("+").split("=", 1)
        if key.startswith("env.slidevqa."):
            config[key.removeprefix("env.slidevqa.")] = yaml.safe_load(value)
    config["reward_fn_path"] = str(vra / "rl/reward_judge.py")
    env = SlideVQAMultiProcessEnv(env_config={"max_steps": 10, "slidevqa": config})
    assert env.coverage_w == 0.2 and env.full_coverage_w == 0.3 and env.grounding_w == 0.5
    assert env.coverage_mode == "fraction" and env.cmp_w == 0 and env.access_cost_w == 0
    rows = {}
    for split in ("train", "val"):
        path = root / f"data/slidevqa/rl_set_1200_kf/{split}.parquet"
        rows[split] = pq.read_metadata(path).num_rows
        assert rows[split] > 0, str(path)
    start = time.monotonic()
    retriever = ColQwenRetriever(
        model_path=str(root / "models/colqwen2.5-v0.1"),
        index_path=root / "data/slidevqa/index/val_colqwen",
        device="cuda:0", dtype="bfloat16", attn_implementation="flash_attention_2",
        bank_device="cpu",
    )
    result = retriever.search("What is the main topic of this presentation?", top_k=1)
    assert len(result) == 1
    image, label = result[0]
    assert image.width > 0 and image.height > 0 and label
    report = {
        "dataset_rows": rows,
        "actual_verl_env_module": type(env).__module__,
        "reward_config_verified": True,
        "retrieval_index_entries": len(retriever.entries),
        "gpu_retrieval_and_image_load_passed": True,
        "retrieved_page_label": label,
        "retrieved_image_size": list(image.size),
        "retrieval_load_and_query_seconds": round(time.monotonic() - start, 2),
        "gpu": torch.cuda.get_device_name(0),
        "visible_gpus": torch.cuda.device_count(),
        "external_judge_requests": 0,
        "four_gpu_nccl_validated": False,
        "policy_training_validated": False,
        "validation_scope": "runtime and migrated asset usability, not answer accuracy",
    }
    (root / "preparation/runtime_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
