from __future__ import annotations

import json
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class IndexEntry:
    image_path: str
    page_label: str


class Retriever:
    def __init__(
        self,
        model_path: str = "Alibaba-NLP/GVE-7B",
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

    def search(self, query: str, top_k: int = 1) -> list[tuple[Image.Image, str]]:
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
        top_indices = np.argsort(-scores)[:top_k]

        results: list[tuple[Image.Image, str]] = []
        for idx in top_indices:
            entry = self.entries[int(idx)]
            image_path = Path(entry.image_path)
            if not image_path.exists():
                continue
            with Image.open(image_path) as image:
                results.append((image.convert("RGB").copy(), entry.page_label))
        return results

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
        model_path: str = "Alibaba-NLP/GVE-7B",
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
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


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
