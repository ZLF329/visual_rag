#!/usr/bin/env python
"""Encode every page image under --pages-dir with ColQwen2.5 -> ColQwenRetriever index:

  OUT/mv_embeddings.pt   list of per-page (seq_len, 128) fp16 tensors (padding tokens dropped)
  OUT/filenames.json     [{image_path, page_label}] in the same order; label = "<deck>/page_NN"
  OUT/meta.json          model, attention implementation, token-length stats

Attention: flash attention, as colpali-engine recommends. flash-attn has no wheel for torch 2.13 /
CUDA 13, so the default is the prebuilt Hub kernel kernels-community/flash-attn3 (stable-ABI build,
Hopper), loaded by transformers through the `kernels` package (0.16.x for transformers 5.17; set
HF_ENDPOINT=https://hf-mirror.com where the Hub is blocked). sdpa is ~2x slower here: Qwen2.5-VL's
vision tower then runs one SDPA call per attention window.

usage: python scripts/build_colqwen_index.py --pages-dir OUT/pages/train --output OUT/index/train_colqwen \
           --model /path/to/colqwen2.5-v0.1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from src.retriever import discover_images, page_label_from_path


class Pages(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Image.Image:
        with Image.open(self.paths[i]) as im:
            return im.convert("RGB").copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--attn", default="kernels-community/flash-attn3", help="attn_implementation (e.g. sdpa)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="encode only the first N pages (speed/accuracy checks)")
    ap.add_argument("--max-visual-tokens", type=int, default=0,
                    help="cap tokens per page (colpali-engine's max_num_visual_tokens); 0 = processor default "
                         "(12.8 MP, fine for slides; high-res document pages need a cap or the vision tower OOMs)")
    args = ap.parse_args()

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    paths = discover_images(args.pages_dir)
    if not paths:
        raise FileNotFoundError(f"no images under {args.pages_dir}")
    if args.limit:
        paths = paths[: args.limit]
    print(f"[index] {len(paths)} pages, attention = {args.attn}", flush=True)
    model = ColQwen2_5.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=args.device,
                                       attn_implementation=args.attn).eval()
    cap = {"max_num_visual_tokens": args.max_visual_tokens} if args.max_visual_tokens else {}
    proc = ColQwen2_5_Processor.from_pretrained(args.model, **cap)
    loader = torch.utils.data.DataLoader(Pages(paths), batch_size=args.batch_size, num_workers=args.num_workers,
                                         collate_fn=proc.process_images)
    embs: list[torch.Tensor] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(model.device)
            out = model(**batch)
            mask = batch["attention_mask"].bool()
            embs.extend(out[i][mask[i]].to("cpu", torch.float16) for i in range(out.shape[0]))
            if len(embs) % (args.batch_size * 50) < args.batch_size:
                rate = len(embs) / (time.time() - t0)
                print(f"[index] {len(embs)}/{len(paths)} pages  {rate:.1f} pages/s", flush=True)

    lens = torch.tensor([e.shape[0] for e in embs], dtype=torch.float32)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embs, out_dir / "mv_embeddings.pt")
    entries = [{"image_path": str(p), "page_label": page_label_from_path(p)} for p in paths]
    (out_dir / "filenames.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {"model": args.model, "attn": args.attn, "max_visual_tokens": args.max_visual_tokens or None,
            "pages": len(embs), "tokens_mean": float(lens.mean()),
            "tokens_max": int(lens.max()), "tokens_min": int(lens.min()), "seconds": round(time.time() - t0, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[index] done", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
