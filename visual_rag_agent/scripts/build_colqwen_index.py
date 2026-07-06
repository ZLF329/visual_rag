from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


def norm_entries(raw):
    out = []
    for it in raw:
        if isinstance(it, str):
            out.append({"image_path": it, "page_label": Path(it).stem})
        else:
            out.append({
                "image_path": str(it["image_path"]),
                "page_label": str(it.get("page_label") or Path(it["image_path"]).stem),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/colqwen2.5-v0.1")
    ap.add_argument("--source-index", required=True, help="dir with filenames.json to mirror coverage+labels")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype, torch.bfloat16)
    entries = norm_entries(json.load(open(Path(args.source_index) / "filenames.json")))
    if args.limit:
        entries = entries[: args.limit]
    print(f"loading ColQwen from {args.model} ...", flush=True)
    model = ColQwen2_5.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device).eval()
    proc = ColQwen2_5_Processor.from_pretrained(args.model)
    print(f"embedding {len(entries)} pages (batch={args.batch_size}) ...", flush=True)

    embs = [None] * len(entries)
    bs = args.batch_size
    done = 0
    for start in range(0, len(entries), bs):
        chunk = entries[start : start + bs]
        imgs, idxs = [], []
        for k, e in enumerate(chunk):
            p = Path(e["image_path"])
            if not p.exists():
                continue
            with Image.open(p) as im:
                imgs.append(im.convert("RGB").copy())
            idxs.append(start + k)
        if not imgs:
            continue
        batch = proc.process_images(imgs).to(model.device)
        with torch.no_grad():
            out = model(**batch)            # (B, L, 128) padded
        mask = batch["attention_mask"].bool()
        for bi, gi in enumerate(idxs):
            valid = out[bi][mask[bi]]        # (L_i, 128) unpadded
            embs[gi] = valid.to(torch.float16).cpu().contiguous()
        done += len(idxs)
        if start // bs % 50 == 0:
            print(f"  {done}/{len(entries)}", flush=True)

    final_embs, final_entries, miss = [], [], 0
    for e, t in zip(entries, embs):
        if t is None:
            miss += 1
            continue
        final_embs.append(t)
        final_entries.append(e)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(final_embs, out / "mv_embeddings.pt")
    json.dump(final_entries, open(out / "filenames.json", "w"), ensure_ascii=False)
    print(f"INDEX_DONE pages={len(final_embs)} missing={miss} out={out}", flush=True)


if __name__ == "__main__":
    main()
