"""Optional image preprocessing.

A/B finding on the benchmark document (13-page scanned chemistry pamphlet,
2026-09): raw color images BEAT grayscale/contrast/sharpen -- preprocessing
destroyed a real content line, produced letter-spam, and was slower.
Preprocessing therefore defaults to OFF everywhere; keep it off unless you
have re-run the A/B on your own corpus and documented a win.
"""

from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter


def preprocess_image(
    image: Image.Image,
    grayscale: bool = False,
    contrast: float = 1.0,
    sharpen: bool = False,
    scale: float = 1.0,
) -> Image.Image:
    image = image.convert("L") if grayscale else image.convert("RGB")

    if scale > 1.0:
        width, height = image.size
        image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)

    if sharpen:
        image = image.filter(ImageFilter.SHARPEN)

    return image
