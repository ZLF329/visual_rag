from __future__ import annotations

import math
import re

from PIL import Image, ImageDraw, ImageFont


def resize_to_pixel_count(
    img: Image.Image,
    target_pixels: int,
    *,
    allow_upscale: bool = False,
) -> Image.Image:
    """
    Resize an image to approximately target_pixels while preserving aspect ratio.

    By default the function never upscales. If allow_upscale=True, the image is
    resized to approximately target_pixels even when the input is smaller.
    """
    if target_pixels <= 0:
        raise ValueError("target_pixels must be positive")

    width, height = img.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {img.size}")

    current_pixels = width * height
    if current_pixels <= target_pixels and not allow_upscale:
        return img

    scale = math.sqrt(target_pixels / current_pixels)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return img.resize((new_width, new_height), resampling)


def add_coordinate_grid(img: Image.Image, *, cols: int = 8, rows: int = 8) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    width, height = out.size
    line_color = (255, 0, 0, 70)
    label_bg = (255, 255, 255, 170)
    label_color = (120, 0, 0, 210)
    font = ImageFont.load_default()

    for col in range(1, cols):
        x = round(width * col / cols)
        draw.line([(x, 0), (x, height)], fill=line_color, width=1)
    for row in range(1, rows):
        y = round(height * row / rows)
        draw.line([(0, y), (width, y)], fill=line_color, width=1)

    for row in range(rows):
        for col in range(cols):
            label = f"{chr(ord('A') + col)}{row + 1}"
            x0 = round(width * col / cols)
            y0 = round(height * row / rows)
            draw.rectangle([x0 + 2, y0 + 2, x0 + 26, y0 + 14], fill=label_bg)
            draw.text((x0 + 4, y0 + 3), label, fill=label_color, font=font)
    return out


def highlight_cells(img: Image.Image, cells: list[str]) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    for cell in normalize_cell_labels(cells):
        bounds = cell_bounds(cell, out.size)
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        draw.rectangle([x0, y0, x1, y1], fill=(255, 230, 0, 70), outline=(255, 0, 0, 210), width=4)
    return out


def crop_grid_cells_bbox(
    img: Image.Image,
    cells: list[str],
    *,
    padding_cells: int = 1,
) -> Image.Image:
    """Crop the predicted grid-cell bbox with cell padding clamped to page bounds."""
    parsed_cells = [
        parsed for cell in cells if (parsed := _parse_grid_cell(cell)) is not None
    ]
    if not parsed_cells:
        return img.convert("RGB").copy()

    width, height = img.size
    pad = max(0, int(padding_cells))
    min_col = max(0, min(col for col, _ in parsed_cells) - pad)
    max_col = min(7, max(col for col, _ in parsed_cells) + pad)
    min_row = max(0, min(row for _, row in parsed_cells) - pad)
    max_row = min(7, max(row for _, row in parsed_cells) + pad)

    x0 = round(width * min_col / 8)
    y0 = round(height * min_row / 8)
    x1 = round(width * (max_col + 1) / 8)
    y1 = round(height * (max_row + 1) / 8)
    return img.convert("RGB").crop((x0, y0, x1, y1))


def normalize_cell_labels(cells: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for cell in cells:
        text = str(cell or "").strip().upper()
        if not text:
            continue
        labels = _expand_cell_range(text) if ":" in text else [text]
        for label in labels:
            if _parse_grid_cell(label) is None or label in seen:
                continue
            normalized.append(label)
            seen.add(label)
    return normalized


def cell_bounds(cell: str, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    parsed = _parse_grid_cell(cell)
    if parsed is None:
        return None
    col, row = parsed
    width, height = size
    return (
        round(width * col / 8),
        round(height * row / 8),
        round(width * (col + 1) / 8),
        round(height * (row + 1) / 8),
    )


def _expand_cell_range(text: str) -> list[str]:
    start_text, end_text = text.split(":", 1)
    start = _parse_grid_cell(start_text)
    end = _parse_grid_cell(end_text)
    if start is None or end is None:
        return []
    start_col, start_row = start
    end_col, end_row = end
    min_col, max_col = sorted((start_col, end_col))
    min_row, max_row = sorted((start_row, end_row))
    return [
        f"{chr(ord('A') + col)}{row + 1}"
        for row in range(min_row, max_row + 1)
        for col in range(min_col, max_col + 1)
    ]


def _parse_grid_cell(cell: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"([A-Ha-h])([1-8])", str(cell).strip())
    if not match:
        return None
    return ord(match.group(1).upper()) - ord("A"), int(match.group(2)) - 1
