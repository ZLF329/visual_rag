from __future__ import annotations

import math
import re

from PIL import Image, ImageDraw, ImageFont


def resize_to_pixel_count(img: Image.Image, target_pixels: int) -> Image.Image:
    """
    Resize an image to approximately target_pixels while preserving aspect ratio.

    The function never upscales. If the input already fits the target budget, the
    original image object is returned.
    """
    if target_pixels <= 0:
        raise ValueError("target_pixels must be positive")

    width, height = img.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {img.size}")

    current_pixels = width * height
    if current_pixels <= target_pixels:
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
    width, height = out.size
    for cell in cells:
        parsed = _parse_grid_cell(cell)
        if parsed is None:
            continue
        col, row = parsed
        x0 = round(width * col / 8)
        y0 = round(height * row / 8)
        x1 = round(width * (col + 1) / 8)
        y1 = round(height * (row + 1) / 8)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 230, 0, 70), outline=(255, 0, 0, 210), width=4)
    return out


def _parse_grid_cell(cell: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"([A-Ha-h])([1-8])", str(cell).strip())
    if not match:
        return None
    return ord(match.group(1).upper()) - ord("A"), int(match.group(2)) - 1
