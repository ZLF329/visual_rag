#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.agent import Agent
from src.config import load_config
from src.retriever import Retriever
from src.vlm import VLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the visual RAG agent on one query.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--index", default=None)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-load-models", action="store_true", help="Validate wiring without loading models.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device or config["runtime"]["device"]
    index_path = args.index or config["models"]["retriever"]["index_path"]
    output_root = Path(args.output or config["runtime"]["output_dir"])
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")

    retriever = Retriever(
        model_path=config["models"]["retriever"]["name"],
        index_path=index_path,
        device=device,
        dtype=config["runtime"]["dtype"],
        attn_implementation=config["runtime"].get("attn_implementation"),
        load_model=not args.no_load_models,
    )
    vlm = VLM(
        model_path=config["models"]["vlm"]["name"],
        device=device,
        max_tokens=config["models"]["vlm"]["max_tokens"],
        temperature=config["models"]["vlm"]["temperature"],
        dtype=config["runtime"]["dtype"],
        attn_implementation=config["runtime"].get("attn_implementation"),
        load_model=not args.no_load_models,
    )
    agent = Agent(
        vlm=vlm,
        retriever=retriever,
        top_k=config["agent"]["top_k"],
        max_iters=config["agent"]["max_iters"],
    )
    result = agent.run(args.query, output_dir=run_dir)
    print(json.dumps({"run_dir": str(run_dir), "answer": result["answer"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
