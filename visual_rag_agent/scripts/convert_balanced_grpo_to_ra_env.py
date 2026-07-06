#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the selected SlideVQA GRPO parquet split into verl-agent SlideVQA RA env parquet.")
    parser.add_argument("--input-dir", default="/scratch/punim0614/lifuzhang/visual_rag_agent/data/grpo/visor_slidevqa_balanced_800")
    parser.add_argument("--output-dir", default="/scratch/punim0614/lifuzhang/visual_rag_agent/data/ra_grpo/slidevqa")
    parser.add_argument("--index-dir", default="/scratch/punim0614/lifuzhang/visual_rag_agent/data/indexes/slidevqa_train_balanced_2000")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_labels = load_index_labels(Path(args.index_dir))
    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "index_dir": str(args.index_dir),
        "splits": {},
    }
    for split in ["train", "val"]:
        in_path = input_dir / f"{split}.parquet"
        out_path = output_dir / f"{split}.parquet"
        rows, split_summary = convert_split(in_path, split=split, index_labels=index_labels)
        write_parquet(rows, out_path)
        split_summary["output_file"] = str(out_path)
        summary["splits"][split] = split_summary

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def convert_split(path: Path, *, split: str, index_labels: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    unique_refs: set[str] = set()
    covered_refs: set[str] = set()
    samples_with_any_ref_in_index = 0

    for index, row in df.iterrows():
        extra = dict(row.get("extra_info") or {})
        question = str(extra.get("question") or extract_prompt_question(row.get("prompt")) or "").strip()
        answer = str(extra.get("answer") or extract_ground_truth(row.get("reward_model")) or "").strip()
        deck_name = str(extra.get("deck_name") or "").strip()
        page_labels = normalize_page_labels(extra.get("page_labels"), deck_name=deck_name, evidence_pages=extra.get("evidence_pages"))
        evidence_pages = normalize_pages(extra.get("evidence_pages"))

        unique_refs.update(page_labels)
        sample_covered = [label for label in page_labels if label in index_labels]
        covered_refs.update(sample_covered)
        if sample_covered:
            samples_with_any_ref_in_index += 1

        data_source = f"slidevqa_ra_{split}"
        prompt = [
            {
                "role": "user",
                "content": (
                    "Answer the SlideVQA question by repeatedly choosing exactly one action: "
                    "<search>query</search> or <answer>answer</answer>.\n\n"
                    f"Question: {question}"
                ),
            }
        ]
        clean_extra = json_safe(extra)
        clean_extra.update({"index": int(index), "split": split, "data_source": data_source})
        env_kwargs = {
            "question": question,
            "answer": answer,
            "ground_truth": answer,
            "deck_name": deck_name,
            "evidence_pages": evidence_pages,
            "reference_pages": page_labels,
            "page_labels": page_labels,
            "data_source": data_source,
            "extra_info": clean_extra,
        }
        rows.append(
            {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "visual_document_qa",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": clean_extra,
                "env_kwargs": env_kwargs,
            }
        )

    split_summary = {
        "input_file": str(path),
        "rows": len(rows),
        "unique_reference_pages": len(unique_refs),
        "unique_reference_pages_in_index": len(covered_refs),
        "reference_page_index_coverage": round(len(covered_refs) / max(1, len(unique_refs)), 6),
        "samples_with_any_reference_page_in_index": samples_with_any_ref_in_index,
    }
    return rows, split_summary


def load_index_labels(path: Path) -> set[str]:
    filenames = path / "filenames.json"
    if not filenames.exists():
        return set()
    data = json.loads(filenames.read_text(encoding="utf-8"))
    labels: set[str] = set()
    for item in data:
        if isinstance(item, dict) and item.get("page_label"):
            labels.add(str(item["page_label"]))
        elif isinstance(item, str):
            labels.add(item)
    return labels


def extract_prompt_question(prompt: Any) -> str:
    if isinstance(prompt, list) and prompt:
        content = prompt[0].get("content") if isinstance(prompt[0], dict) else ""
        marker = "Question:"
        if isinstance(content, str) and marker in content:
            return content.split(marker, 1)[1].strip()
    return ""


def extract_ground_truth(reward_model: Any) -> str:
    if isinstance(reward_model, dict):
        return str(reward_model.get("ground_truth") or "")
    return ""


def normalize_page_labels(value: Any, *, deck_name: str, evidence_pages: Any) -> list[str]:
    labels = [str(x) for x in as_list(value) if str(x)]
    if labels:
        return labels
    pages = normalize_pages(evidence_pages)
    return [f"{deck_name}/page_{page:02d}" for page in pages if deck_name]


def normalize_pages(value: Any) -> list[int]:
    pages: list[int] = []
    for item in as_list(value):
        try:
            page = int(item)
        except (TypeError, ValueError):
            continue
        if page > 0 and page not in pages:
            pages.append(page)
    return pages


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


if __name__ == "__main__":
    main()
