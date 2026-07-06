#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SlideVQA PDFs into page images.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pdf-root", default=None)
    parser.add_argument("--pdf-zip", default=None)
    parser.add_argument("--eval-jsonl", default=None)
    parser.add_argument("--dpi", type=int, default=144)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pdf_root = prepare_pdf_root(output, args.pdf_root, args.pdf_zip)
    if pdf_root is None:
        raise ValueError("provide --pdf-root or --pdf-zip")

    render_pdfs(pdf_root, output / "pages", dpi=args.dpi)
    if args.eval_jsonl:
        shutil.copy2(args.eval_jsonl, output / Path(args.eval_jsonl).name)


def prepare_pdf_root(output: Path, pdf_root: str | None, pdf_zip: str | None) -> Path | None:
    if pdf_root:
        return Path(pdf_root)
    if not pdf_zip:
        return None
    extracted = output / "pdfs"
    marker = extracted / ".extracted"
    if marker.exists():
        return extracted
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(pdf_zip, "r") as zf:
        zf.extractall(extracted)
    marker.write_text("ok", encoding="utf-8")
    return extracted


def render_pdfs(pdf_root: Path, image_root: Path, *, dpi: int) -> None:
    import fitz

    image_root.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for pdf_path in sorted(pdf_root.rglob("*.pdf")):
        doc_id = pdf_path.stem
        doc_out = image_root / doc_id
        doc_out.mkdir(parents=True, exist_ok=True)
        with fitz.open(pdf_path) as doc:
            for page_idx, page in enumerate(doc, start=1):
                out_path = doc_out / f"page_{page_idx:04d}.png"
                if out_path.exists():
                    continue
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pixmap.save(out_path)


if __name__ == "__main__":
    main()
