#!/usr/bin/env python3
"""Download public retriever weights only when their hashes match the original run."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

MODELS = {
    "colqwen2.5-base": {
        "revision": "92908120384b7a2110c5beda3ab29cbdb2c08e49",
        "files": {
            "model-00001-of-00002.safetensors": "6b45c7afe391b4d9cc49f1ed3f6976f4a25ed40aa2165ed2ae118ff549355985",
            "model-00002-of-00002.safetensors": "32c18f3342ed82c900e02a01b0a6fe7c6aef16eb8ae9a4a83547fbe6e057d512",
        },
    },
    "colqwen2.5-v0.1": {
        "revision": "4379ecfd4c9446836ad0ef4dd7390b1c748f68d8",
        "files": {
            "adapter_model.safetensors": "955d771b5cae99cdbac82d8960dd1731ff92a0d89a2f15475e418fb5dbae44ed",
            "1_Dense/model.safetensors": "157bd2335707a981a060a6c6bfe7f90458c196f772c1c223cc07a379a83dcaa8",
        },
    },
}


def download(item):
    model, filename, revision, expected = item
    dest = Path(os.environ.get("WORK_ROOT", "/workspace")) / "models" / model
    path = Path(hf_hub_download(f"vidore/{model}", filename, revision=revision, local_dir=dest))
    return verify(model, filename, expected, path)


def verify(model, filename, expected, path=None):
    if path is None:
        path = Path(os.environ.get("WORK_ROOT", "/workspace")) / "models" / model / filename
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"Source weight checksum mismatch: {model}/{filename}")
    result = {"model": model, "file": filename, "bytes": path.stat().st_size, "sha256": actual}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    tasks = []
    for model, spec in MODELS.items():
        if not args.verify_only:
            info = HfApi().model_info(f"vidore/{model}", revision=spec["revision"], files_metadata=True)
            files = {p.rfilename: p for p in info.siblings}
        for filename, expected in spec["files"].items():
            if not args.verify_only:
                assert files[filename].lfs.sha256 == expected, f"Official weight differs: {filename}"
            tasks.append((model, filename, spec["revision"], expected))
    with ThreadPoolExecutor(max_workers=3) as pool:
        if args.verify_only:
            results = list(pool.map(lambda t: verify(t[0], t[1], t[3]), tasks))
        else:
            results = list(pool.map(download, tasks))
    out = Path(os.environ.get("WORK_ROOT", "/workspace")) / "preparation/retriever_weights_verified.json"
    out.write_text(json.dumps(results, indent=2))
    print("RETRIEVER_WEIGHTS_MATCH_SOURCE", flush=True)


if __name__ == "__main__":
    main()
