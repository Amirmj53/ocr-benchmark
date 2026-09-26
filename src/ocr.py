"""RapidOCR engine wrapper.

Engine choice is fixed by project benchmarks: RapidOCR (ONNX Runtime) with
the PP-OCRv5 Arabic mobile recognition model is significantly faster than
PaddleOCR on this machine while giving comparable quality. Do not swap the
engine without a documented benchmark win.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

from models import OCRBlock


def create_ocr_engine() -> RapidOCR:
    return RapidOCR(
        params={
            "Rec.lang_type": LangRec.ARABIC,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
        }
    )


class OCREngine:
    """Thin stateful wrapper: loads models once, OCRs PIL images."""

    def __init__(self) -> None:
        self._engine = create_ocr_engine()

    def run_image(self, image) -> list[OCRBlock]:
        """Run OCR on a PIL image, returning raw blocks (unfiltered)."""
        array = np.asarray(image.convert("RGB"))
        start = time.perf_counter()
        result = self._engine(array)
        elapsed = time.perf_counter() - start

        blocks: list[OCRBlock] = []
        if result is None or result.boxes is None:
            return blocks

        page_height = float(array.shape[0])
        for box, text, score in zip(result.boxes, result.txts, result.scores):
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            blocks.append(
                OCRBlock(
                    text=str(text).strip(),
                    score=float(score),
                    left=min(xs),
                    top=min(ys),
                    right=max(xs),
                    bottom=max(ys),
                    page_number=0,  # set by caller
                )
            )

        # Attach real page number later; keep helper for footer zones.
        self._last_image_height = page_height
        return blocks

    def run_page(self, image, page_number: int) -> tuple[list[OCRBlock], float]:
        """Run OCR on one page image. Returns (blocks, seconds)."""
        start = time.perf_counter()
        blocks = self.run_image(image)
        elapsed = time.perf_counter() - start
        for block in blocks:
            block.page_number = page_number
        return blocks, elapsed

    # Kept for the older file-path based tests.
    def run_path(self, image_path: str):
        return self._engine(str(image_path))
