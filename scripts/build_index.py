#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.retriever import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple numpy image index.")
    parser.add_argument("--corpus", required=True, help="Directory containing rendered page images.")
    parser.add_argument("--output", required=True, help="Output index directory.")
    parser.add_argument("--model", default="Alibaba-NLP/GVE-7B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    build_index(
        corpus=args.corpus,
        output=args.output,
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
