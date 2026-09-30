"""RapidOCR engine wrapper with a process-wide singleton.

Engine choice is fixed by project benchmarks: RapidOCR (ONNX Runtime) with
the PP-OCRv5 Arabic mobile recognition model is significantly faster than
PaddleOCR on the reference machine while giving comparable quality. Do not
swap the engine without a documented benchmark win.

The models are loaded ONCE per process and reused for every page. Loading
per page would dominate runtime and spike RAM (~1 GB peak), so the engine
is never created per job or per page: call get_ocr_engine().
"""

from __future__ import annotations

import threading
import time

import numpy as np
from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

from darsio_ingest.models import OCRBlock


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
        return blocks

    def run_page(self, image, page_number: int) -> tuple[list[OCRBlock], float]:
        """Run OCR on one page image. Returns (blocks, seconds)."""
        start = time.perf_counter()
        blocks = self.run_image(image)
        elapsed = time.perf_counter() - start
        for block in blocks:
            block.page_number = page_number
        return blocks, elapsed


_ENGINE_LOCK = threading.Lock()
_ENGINE: OCREngine | None = None


def get_ocr_engine() -> OCREngine:
    """Process-wide engine accessor. Loads models on first use only.

    The OCR engine is NOT thread-safe; keep all OCR on one worker thread
    (the documented deployment shape is a single-worker queue).
    """
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = OCREngine()
    return _ENGINE


def reset_ocr_engine() -> None:
    """Drop the singleton (tests / explicit memory reclaim only)."""
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = None
