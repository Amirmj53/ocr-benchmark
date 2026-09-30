"""Text-extraction path: healthy-text pages, no OCR.

Extracts the PDF text layer with pymupdf ("dict" mode), builds lines from
spans and reuses the SAME layout/normalization machinery as the OCR path so
chunks from both paths behave identically downstream.

What this path does NOT do: render images, load OCR models, or decide page
ranges. Method choice lives in pipeline.py, driven by diagnose.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from darsio_ingest.layout import build_page_result
from darsio_ingest.models import OCRBlock, PageResult
from darsio_ingest.normalize import normalize_text

# Score assigned to extracted-text lines. The PDF text layer is exact by
# definition, so text-path content is always trusted (never filtered by the
# OCR confidence thresholds).
TEXT_SCORE = 1.0


def extract_page_text(path: str | Path, page_number: int) -> str:
    """Extract one page's text layer in reading order, normalized."""
    doc = pymupdf.open(path)
    try:
        page = doc[page_number - 1]
        blocks = page.get_text("blocks", sort=True) or []
    finally:
        doc.close()

    lines = [
        normalize_text(str(b[4]).strip()) for b in blocks if (b[4] or "").strip()
    ]
    return "\n\n".join(lines)


def build_text_page_result(
    page: pymupdf.Page, page_number: int, status: str = ""
) -> PageResult:
    """Turn one PDF page's text layer into a structured PageResult.

    Uses the same layout pipeline as OCR output (line grouping, column
    detection, region building) by wrapping extracted spans as OCRBlocks
    with perfect score=1.0.
    """
    raw = page.get_text("dict", sort=True) or {}
    blocks: list[OCRBlock] = []

    for pdf_block in raw.get("blocks", []):
        if pdf_block.get("type") != 0:
            continue  # image blocks belong to the OCR path (if requested)
        for line_dict in pdf_block.get("lines", []):
            spans = line_dict.get("spans", [])
            line_text = "".join(
                span.get("text", "") for span in spans
            ).strip()
            if not line_text:
                continue
            bbox = line_dict.get("bbox", pdf_block.get("bbox", (0, 0, 0, 0)))
            blocks.append(
                OCRBlock(
                    text=normalize_text(line_text),
                    score=TEXT_SCORE,
                    page_number=page_number,
                    left=float(bbox[0]),
                    top=float(bbox[1]),
                    right=float(bbox[2]),
                    bottom=float(bbox[3]),
                )
            )

    return build_page_result(
        page_number=page_number,
        width=float(page.rect.width),
        height=float(page.rect.height),
        blocks=blocks,
        source_method="text",
        status=status,
    )


def build_text_page_results(
    path: str | Path,
    pages: list[int],
    status_per_page: dict[int, str] | None = None,
) -> list[PageResult]:
    """Build structured PageResults for selected pages via the text path.

    Opens the document once for the whole batch. Out-of-range page numbers
    are skipped.
    """
    status_per_page = status_per_page or {}
    doc = pymupdf.open(path)
    try:
        results: list[PageResult] = []
        for number in pages:
            if not 1 <= number <= doc.page_count:
                continue
            page = doc[number - 1]
            results.append(
                build_text_page_result(
                    page, number, status=status_per_page.get(number, "")
                )
            )
        return results
    finally:
        doc.close()
