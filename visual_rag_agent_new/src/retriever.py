from __future__ import annotations

import json
import importlib.util
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class IndexEntry:
    image_path: str
    page_label: str


class Retriever:
    def __init__(
        self,
        model_path: str = "/root/autodl-tmp/models/Qwen3-VL-Embedding-8B",
        index_path: str | Path = "data/indexes/slidevqa",
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        load_model: bool = True,
    ) -> None:
        self.model_path = model_path
        self.index_path = Path(index_path)
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.embeddings: np.ndarray | None = None
        self.entries: list[IndexEntry] = []
        self.embedder: GVEEmbedder | None = None
        self._load_index()
        if load_model:
            self.embedder = GVEEmbedder(
                model_path=model_path,
                device=device,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )

    def search(
        self,
        query: str,
        top_k: int = 1,
        deck_name: str | None = None,
    ) -> list[tuple[Image.Image, str]]:
        if self.embeddings is None:
            raise RuntimeError("retriever index is not loaded")
        if self.embedder is None:
            self.embedder = GVEEmbedder(
                model_path=self.model_path,
                device=self.device,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
            )

        query_embedding = self.embedder.embed_texts([query])[0]
        query_embedding = normalize_rows(query_embedding[None, :])[0]
        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(-scores)
        prefix = f"{deck_name}/" if deck_name else ""

        results: list[tuple[Image.Image, str]] = []
        for idx in ranked_indices:
            entry = self.entries[int(idx)]
            if prefix and not entry.page_label.startswith(prefix):
                continue
            image_path = Path(entry.image_path)
            if not image_path.exists():
                continue
            with Image.open(image_path) as image:
                results.append((image.convert("RGB").copy(), entry.page_label))
            if len(results) >= top_k:
                break
        return results

    def search_labels(
        self,
        query: str,
        top_k: int = 1,
        deck_name: str | None = None,
    ) -> list[str]:
        if self.embeddings is None:
            raise RuntimeError("retriever index is not loaded")
        if self.embedder is None:
            self.embedder = GVEEmbedder(
                model_path=self.model_path,
                device=self.device,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
            )

        query_embedding = self.embedder.embed_texts([query])[0]
        query_embedding = normalize_rows(query_embedding[None, :])[0]
        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(-scores)
        prefix = f"{deck_name}/" if deck_name else ""

        labels: list[str] = []
        for idx in ranked_indices:
            entry = self.entries[int(idx)]
            if prefix and not entry.page_label.startswith(prefix):
                continue
            labels.append(entry.page_label)
            if len(labels) >= top_k:
                break
        return labels

    def _load_index(self) -> None:
        embeddings_path = self.index_path / "embeddings.npy"
        filenames_path = self.index_path / "filenames.json"
        if not embeddings_path.exists() or not filenames_path.exists():
            return

        embeddings = np.load(embeddings_path).astype("float32")
        self.embeddings = normalize_rows(embeddings)
        with filenames_path.open("r", encoding="utf-8") as f:
            raw_entries = json.load(f)

        self.entries = []
        for item in raw_entries:
            if isinstance(item, str):
                self.entries.append(IndexEntry(image_path=item, page_label=Path(item).stem))
            else:
                self.entries.append(
                    IndexEntry(
                        image_path=str(item["image_path"]),
                        page_label=str(item.get("page_label") or Path(item["image_path"]).stem),
                    )
                )


class GVEEmbedder:
    def __init__(
        self,
        model_path: str = "/root/autodl-tmp/models/Qwen3-VL-Embedding-8B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.processor: Any | None = None
        self.model: Any | None = None
        self._load_model()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        if hasattr(self.model, "process"):
            return self._to_numpy(
                self.model.process(
                    [
                        {
                            "text": text,
                            "instruction": "Represent this query for visual document page retrieval.",
                        }
                        for text in texts
                    ]
                )
            )
        if hasattr(self.model, "encode_text"):
            return self._to_numpy(self.model.encode_text(texts))
        if hasattr(self.model, "encode"):
            try:
                return self._to_numpy(self.model.encode(texts))
            except TypeError:
                pass
        return self._embed_with_processor(texts=texts, images=None)

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        self._ensure_loaded()
        if hasattr(self.model, "process"):
            return self._to_numpy(
                self.model.process(
                    [
                        {
                            "image": image,
                            "instruction": "Represent this document page image for retrieval.",
                        }
                        for image in images
                    ]
                )
            )
        if hasattr(self.model, "encode_image"):
            return self._to_numpy(self.model.encode_image(images))
        if hasattr(self.model, "encode"):
            try:
                return self._to_numpy(self.model.encode(images))
            except TypeError:
                pass
        return self._embed_with_processor(texts=None, images=images)

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        dtype = getattr(torch, self.dtype, torch.bfloat16)
        local_embedding_script = Path(self.model_path) / "scripts" / "qwen3_vl_embedding.py"
        if local_embedding_script.exists():
            module = load_module_from_path("qwen3_vl_embedding_local", local_embedding_script)
            kwargs: dict[str, Any] = {"torch_dtype": dtype}
            if self.attn_implementation:
                kwargs["attn_implementation"] = self.attn_implementation
            self.model = module.Qwen3VLEmbedder(self.model_path, **kwargs)
            self.processor = getattr(self.model, "processor", None)
            return

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        self.model = AutoModel.from_pretrained(self.model_path, **kwargs)
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def _embed_with_processor(
        self,
        *,
        texts: list[str] | None,
        images: list[Image.Image] | None,
    ) -> np.ndarray:
        import torch

        kwargs: dict[str, Any] = {"return_tensors": "pt", "padding": True}
        if texts is not None:
            kwargs["text"] = texts
        if images is not None:
            kwargs["images"] = images
        inputs = self.processor(**kwargs)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        with torch.no_grad():
            outputs = self.model(**inputs)
        pooled = pool_model_output(outputs, inputs.get("attention_mask"))
        return self._to_numpy(pooled)

    def _ensure_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("embedder model is not loaded")
        if self.processor is None and not hasattr(self.model, "process"):
            raise RuntimeError("embedder processor is not loaded")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return normalize_rows(value.astype("float32"))
        if isinstance(value, (list, tuple)):
            value = value[0] if len(value) == 1 else value
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return normalize_rows(np.asarray(value, dtype="float32"))


def pool_model_output(outputs: Any, attention_mask: Any | None) -> Any:
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if isinstance(outputs, dict) and outputs.get("pooler_output") is not None:
        return outputs["pooler_output"]
    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype="float32")
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_images(corpus: str | Path) -> list[Path]:
    root = Path(corpus)
    import os as _os
    out=[]
    for _dp,_dn,_fs in _os.walk(root, followlinks=True):
        for _f in _fs:
            _p=Path(_dp)/_f
            if _p.suffix.lower() in IMAGE_EXTENSIONS:
                out.append(_p)
    return sorted(out)


def build_index(
    *,
    corpus: str | Path,
    output: str | Path,
    model_path: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    batch_size: int = 8,
) -> None:
    image_paths = discover_images(corpus)
    if not image_paths:
        raise FileNotFoundError(f"no images found under {corpus}")

    embedder = GVEEmbedder(
        model_path=model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    all_embeddings: list[np.ndarray] = []
    entries: list[dict[str, str]] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images: list[Image.Image] = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        all_embeddings.append(embedder.embed_images(images))
        entries.extend(
            {
                "image_path": str(path),
                "page_label": page_label_from_path(path),
            }
            for path in batch_paths
        )

    embeddings = normalize_rows(np.concatenate(all_embeddings, axis=0)).astype("float32")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", embeddings)
    with (output / "filenames.json").open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def page_label_from_path(path: Path) -> str:
    parent = path.parent.name
    stem = path.stem
    return f"{parent}/{stem}" if parent else stem


class ColQwenRetriever:
    """Multi-vector (ColPali late-interaction) retriever using ColQwen2.5.

    Drop-in for Retriever (search / search_labels). Index = list of per-page (seq_len, 128) fp16
    tensors in mv_embeddings.pt + filenames.json (scripts/build_colqwen_index.py). At load the pages
    are packed into a length-sorted fp16 PackedBank. A query is scored on the encoder's GPU against
    the whole bank (flat corpus) or the pages of one deck. bank_device="cpu" keeps the bank in pinned
    RAM and streams its chunks to the GPU (~10 GB per flat query over PCIe), for when the GPU is
    shared with a vLLM server.
    """

    def __init__(
        self,
        model_path: str = "/root/autodl-tmp/models/colqwen2.5-v0.1",
        index_path: str | Path = "data/indexes/slidevqa_test_colqwen",
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        load_model: bool = True,
        bank_device: str | None = None,
        chunk_size: int = 1024,
    ) -> None:
        self.model_path = model_path
        self.index_path = Path(index_path)
        self.device = "cuda:0" if device == "cuda" else device
        self.bank_device = self.device if bank_device in (None, "", "auto") else bank_device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.chunk_size = chunk_size
        self.entries: list[IndexEntry] = []
        self.bank: Any | None = None
        self.deck_to_indices: dict[str, list[int]] = {}
        self.model: Any | None = None
        self.proc: Any | None = None
        self._load_index()
        if load_model:
            self._load_model()

    def _load_index(self) -> None:
        import torch

        emb_path = self.index_path / "mv_embeddings.pt"
        fn_path = self.index_path / "filenames.json"
        if not emb_path.exists() or not fn_path.exists():
            return
        page_embs = torch.load(emb_path, map_location="cpu")
        with fn_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if len(raw) != len(page_embs):
            raise ValueError(f"{fn_path}: {len(raw)} entries but {len(page_embs)} embeddings")
        self.entries = [
            IndexEntry(image_path=item, page_label=Path(item).stem) if isinstance(item, str)
            else IndexEntry(image_path=str(item["image_path"]),
                            page_label=str(item.get("page_label") or Path(item["image_path"]).stem))
            for item in raw
        ]
        self.deck_to_indices = {}
        for i, e in enumerate(self.entries):
            deck = e.page_label.rsplit("/", 1)[0] if "/" in e.page_label else ""
            self.deck_to_indices.setdefault(deck, []).append(i)
        self.bank = pack_bank(page_embs, device=self.bank_device, chunk_size=self.chunk_size)

    def _load_model(self) -> None:
        import torch
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        dtype = getattr(torch, self.dtype, torch.bfloat16)
        kwargs: dict[str, Any] = {"torch_dtype": dtype, "device_map": self.device}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        self.model = ColQwen2_5.from_pretrained(self.model_path, **kwargs).eval()
        self.proc = ColQwen2_5_Processor.from_pretrained(self.model_path)

    def _candidates(self, deck_name: str | None) -> list[int] | None:
        """Row ids of the deck, or None for the whole corpus."""
        if not deck_name:
            return None
        if deck_name not in self.deck_to_indices:
            raise KeyError(f"deck {deck_name!r} is not in the index {self.index_path}")
        return self.deck_to_indices[deck_name]

    def _rank(self, query: str, deck_name: str | None, top_k: int) -> list[int]:
        import torch

        if self.bank is None:
            raise RuntimeError("colqwen index is not loaded")
        if self.model is None:
            self._load_model()
        batch_q = self.proc.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            q = self.model(**batch_q)[0]
        q = q[batch_q["attention_mask"][0].bool()]
        cand = self._candidates(deck_name)
        scores = maxsim_scores(q, self.bank, None if cand is None else torch.tensor(cand), device=self.model.device)
        order = torch.topk(scores, k=min(top_k, scores.numel())).indices.tolist()
        return [int(i) for i in order] if cand is None else [cand[int(i)] for i in order]

    def search(
        self,
        query: str,
        top_k: int = 1,
        deck_name: str | None = None,
    ) -> list[tuple[Image.Image, str]]:
        results: list[tuple[Image.Image, str]] = []
        for idx in self._rank(query, deck_name, top_k):
            entry = self.entries[idx]
            with Image.open(entry.image_path) as image:
                results.append((image.convert("RGB").copy(), entry.page_label))
        return results

    def search_labels(
        self,
        query: str,
        top_k: int = 1,
        deck_name: str | None = None,
    ) -> list[str]:
        return [self.entries[i].page_label for i in self._rank(query, deck_name, top_k)]


class PackedBank:
    """Per-page (L_i, D) multi-vectors packed for MaxSim: pages sorted by length and grouped in
    chunks, each chunk padded to its own longest page (with copies of a page's first vector, which
    leaves MaxSim unchanged). fp16; a CPU bank is pinned so chunks stream to the GPU."""

    def __init__(self, page_embs: list[Any], *, device: str, chunk_size: int = 1024) -> None:
        import torch

        lengths = torch.tensor([int(e.shape[0]) for e in page_embs])
        n, dim = len(page_embs), int(page_embs[0].shape[1])
        self.order = torch.argsort(lengths, stable=True)            # chunk row -> page id
        self.where = torch.empty_like(self.order)
        self.where[self.order] = torch.arange(n)                     # page id -> packed position
        self.chunk_size = chunk_size
        pin = str(device) == "cpu" and torch.cuda.is_available()
        self.chunks: list[Any] = []
        for start in range(0, n, chunk_size):
            ids = self.order[start:start + chunk_size].tolist()
            chunk = torch.empty((len(ids), int(lengths[ids].max()), dim), dtype=torch.float16, device=device,
                                pin_memory=pin)
            for j, i in enumerate(ids):
                k = int(page_embs[i].shape[0])
                chunk[j, :k] = page_embs[i].to(device=device, dtype=torch.float16)
                chunk[j, k:] = chunk[j, 0]
            self.chunks.append(chunk)

    def __len__(self) -> int:
        return int(self.order.numel())

    def gather(self, rows: Any) -> Any:
        """Pages `rows` as one padded (len(rows), L, D) tensor on the bank's device."""
        import torch

        pages = []
        for r in rows.tolist():
            pos = int(self.where[r])
            pages.append(self.chunks[pos // self.chunk_size][pos % self.chunk_size])
        length = max(p.shape[0] for p in pages)
        return torch.stack([torch.cat([p, p[:1].expand(length - p.shape[0], -1)]) for p in pages])


def pack_bank(page_embs: list[Any], *, device: str, chunk_size: int = 1024) -> PackedBank:
    return PackedBank(page_embs, device=device, chunk_size=chunk_size)


def _maxsim(q: Any, docs: Any) -> Any:
    """q (Lq, D), docs (n, L, D) -> (n,) float32; one 2-D GEMM."""
    sim = (docs.reshape(-1, docs.shape[-1]) @ q.T).view(docs.shape[0], docs.shape[1], -1)
    return sim.amax(dim=1).float().sum(dim=1)


def maxsim_scores(query: Any, bank: PackedBank, rows: Any | None = None, device: Any | None = None) -> Any:
    """ColBERT MaxSim of one query (Lq, D) against every page (rows=None, in page-id order) or the
    pages `rows` -> float32 scores on `device` (default: the bank's). Chunks of a RAM bank are
    copied to `device`; fp16 on GPU, fp32 on CPU."""
    import torch

    device = torch.device(device) if device is not None else bank.chunks[0].device
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    q = query.to(device=device, dtype=dtype)
    if rows is not None:
        return _maxsim(q, bank.gather(rows).to(device=device, dtype=dtype, non_blocking=True))
    out = torch.empty(len(bank), dtype=torch.float32, device=device)
    for c, chunk in enumerate(bank.chunks):
        ids = bank.order[c * bank.chunk_size: c * bank.chunk_size + chunk.shape[0]].to(device)
        out[ids] = _maxsim(q, chunk.to(device=device, dtype=dtype, non_blocking=True))
    return out


class LockedRetriever:
    """Serializes search() so concurrent rollouts can share one GPU retriever."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    def search(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._inner.search(*args, **kwargs)


def make_retriever(model_path: str, index_path: str | Path, *, device: str = "cuda", dtype: str = "bfloat16",
                   attn_implementation: str | None = None, bank_device: str | None = None) -> Any:
    """ColQwenRetriever when the model path names a ColQwen checkpoint, else the GVE Retriever."""
    if "colqwen" in str(model_path).lower():
        return ColQwenRetriever(model_path=model_path, index_path=index_path, device=device, dtype=dtype,
                                attn_implementation=attn_implementation, bank_device=bank_device)
    return Retriever(model_path=model_path, index_path=index_path, device=device, dtype=dtype,
                     attn_implementation=attn_implementation)
