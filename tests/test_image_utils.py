from PIL import Image

from src.image_utils import resize_to_pixel_count


def test_resize_down_preserves_aspect_and_budget() -> None:
    image = Image.new("RGB", (1000, 500), "white")
    resized = resize_to_pixel_count(image, 100_000)
    assert resized.size[0] * resized.size[1] <= 100_000
    assert abs((resized.size[0] / resized.size[1]) - 2.0) < 0.02


def test_resize_does_not_upscale() -> None:
    image = Image.new("RGB", (100, 100), "white")
    resized = resize_to_pixel_count(image, 400_000)
    assert resized is image
