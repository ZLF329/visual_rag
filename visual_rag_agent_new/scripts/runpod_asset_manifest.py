#!/usr/bin/env python3
"""Create or verify a checksum manifest for the explicitly selected migration assets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ASSETS = ["code/verl-agent", "data/slidevqa/rl_set_1200_kf", "data/slidevqa/index",
          "data/slidevqa/pages", "data/slidevqa/test.jsonl",
          "models/colqwen2.5-base", "models/colqwen2.5-v0.1"]
SKIP = {".git", ".venv", "venv", "__pycache__", ".cache", ".env", ".secrets.env"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.verify:
        source = json.loads(args.verify.read_text())
        def check(record):
            rel = Path(record["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe manifest path")
            path = args.root / rel
            if not path.is_file():
                return {"path": str(rel), "error": "missing"}
            elif path.stat().st_size != record["bytes"]:
                return {"path": str(rel), "error": "size"}
            elif digest(path) != record["sha256"]:
                return {"path": str(rel), "error": "sha256"}
            return None
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 64))) as pool:
            errors = [error for error in pool.map(check, source["files"]) if error is not None]
        verified = len(source["files"]) - len(errors)
        result = {"expected_files": len(source["files"]), "verified_files": verified,
                  "error_count": len(errors), "errors": errors[:30], "all_files_match": not errors}
    else:
        files = []
        for name in ASSETS:
            path = args.root / name
            paths = sorted(path.rglob("*")) if path.is_dir() else [path]
            for item in paths:
                rel = item.relative_to(args.root)
                if (set(rel.parts) & SKIP or any(p.endswith(".egg-info") for p in rel.parts)
                        or item.suffix in (".pyc", ".pem") or not item.is_file()):
                    continue
                files.append({"path": str(rel), "bytes": item.stat().st_size, "sha256": digest(item)})
        result = {"files": files, "file_count": len(files), "total_bytes": sum(f["bytes"] for f in files)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}), flush=True)
    if args.verify and result["error_count"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
