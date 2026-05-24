from __future__ import annotations

import math

from PIL import Image


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
