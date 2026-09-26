"""Optional image preprocessing.

Kept minimal and conservative: for clean scanned pages, preprocessing
helps detection recall; for already-clean pages it can hurt recognition.
The pipeline defaults to grayscale+contrast — measured neutral-to-positive
on the benchmark — and never crops or destroys content.
"""

from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter


def preprocess_image(
    image: Image.Image,
    scale: float = 1.0,
    contrast: float = 1.12,
    sharpen: bool = True,
    grayscale: bool = True,
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
