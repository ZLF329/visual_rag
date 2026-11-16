#!/usr/bin/env python3
"""Reassemble the transferred training index and publish only after SHA-256 matches."""
import hashlib
import json
import os
from pathlib import Path


def main():
    root = Path(os.environ.get("WORK_ROOT", "/workspace"))
    chunks = root / "preparation/index_train_chunks"
    manifest = json.loads((chunks / "manifest.json").read_text())
    source = manifest["source"]
    assert source["path"] == "data/slidevqa/index/train_colqwen/mv_embeddings.pt"
    target = root / source["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".assembling")
    h = hashlib.sha256()
    total = 0
    with temporary.open("wb") as output:
        for part in manifest["parts"]:
            assert Path(part["name"]).name == part["name"]
            path = chunks / part["name"]
            assert path.stat().st_size == part["bytes"], str(path)
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                    output.write(block)
                    h.update(block)
                    total += len(block)
        output.flush()
        os.fsync(output.fileno())
    assert total == source["bytes"], "Index size mismatch"
    assert h.hexdigest() == source["sha256"], "Index SHA-256 mismatch"
    os.replace(temporary, target)
    result = {"path": str(target), "bytes": total, "sha256": h.hexdigest(), "matches_source": True}
    (root / "preparation/train_index_assembly.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
