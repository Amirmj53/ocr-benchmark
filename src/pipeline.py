"""End-to-end pipeline: PDF -> render -> OCR -> layout -> normalize -> chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from chunking import chunk_document
from layout import build_page_result, strip_repeated_content
from models import Chunk, DocumentMetadata, DocumentResult, PageResult
from normalize import normalize_text
from ocr import OCREngine
from pdf import DocumentInfo, open_document
from preprocessing import preprocess_image


@dataclass
class PipelineConfig:
    dpi: int = 200
    # A/B measured on the benchmark PDF: raw color beats grayscale/contrast/
    # sharpen (preprocessing destroyed real content lines and recovered none).
    grayscale: bool = False
    contrast: float = 1.0
    sharpen: bool = False
    strip_repeated: bool = True
    keep_noise_blocks: bool = False
    save_debug_images: bool = False
    debug_image_dir: Path = Path("data/output/debug")


@dataclass
class PipelineRun:
    document: DocumentResult
    elapsed_seconds: float
    ocr_seconds: float
    page_seconds: dict[int, float] = field(default_factory=dict)
    repeated_texts: set[str] = field(default_factory=set)


def process_page(
    engine: OCREngine,
    page_input,
    config: PipelineConfig,
) -> tuple[PageResult, float]:
    """OCR + layout for a single rendered page. Returns (page, ocr_seconds)."""
    image = preprocess_image(
        page_input.image,
        grayscale=config.grayscale,
        contrast=config.contrast,
        sharpen=config.sharpen,
    )

    blocks, ocr_seconds = engine.run_page(image, page_input.page_number)

    # Image pixels == PDF points * dpi/72; layout works in pixel space
    # consistently, so pass the pixel size as the page size.
    pixel_width = float(page_input.image.size[0])
    pixel_height = float(page_input.image.size[1])

    page = build_page_result(
        page_number=page_input.page_number,
        width=pixel_width,
        height=pixel_height,
        blocks=blocks,
        debug_keep_noise=config.keep_noise_blocks,
    )

    # Normalize text while geometry is preserved.
    for block in page.blocks:
        block.text = normalize_text(block.text)
    for line in page.lines:
        for block in line.blocks:
            block.text = normalize_text(block.text)
    for region in page.regions:
        for line in region.lines:
            for block in line.blocks:
                block.text = normalize_text(block.text)
        region.heading_text = (
            normalize_text(region.heading_text) if region.heading_text else None
        )
    page.headings = [normalize_text(h) for h in page.headings]
    page.text = normalize_text(page.text)

    return page, ocr_seconds


def run_pipeline(
    pdf_path: str | Path,
    config: PipelineConfig | None = None,
    page_numbers: list[int] | None = None,
) -> PipelineRun:
    """Process a PDF end-to-end and return structured results + chunks."""
    config = config or PipelineConfig()
    from pdf import iter_pages

    info: DocumentInfo = open_document(pdf_path)
    engine = OCREngine()

    start = time.perf_counter()
    pages: list[PageResult] = []
    page_seconds: dict[int, float] = {}
    ocr_total = 0.0

    for page_input in iter_pages(pdf_path, dpi=config.dpi, page_numbers=page_numbers):
        if config.save_debug_images:
            from pdf import save_page_image

            save_page_image(page_input.image, config.debug_image_dir, page_input.page_number)

        page, ocr_seconds = process_page(engine, page_input, config)
        pages.append(page)
        page_seconds[page.page_number] = ocr_seconds
        ocr_total += ocr_seconds
        print(
            f"  page {page.page_number}/{info.page_count}: "
            f"{len(page.blocks)} blocks, {len(page.regions)} regions, "
            f"{ocr_seconds:.1f}s OCR",
            flush=True,
        )

    repeated: set[str] = set()
    if config.strip_repeated and len(pages) >= 4:
        repeated = strip_repeated_content(pages)

    metadata = DocumentMetadata(
        document_id=info.document_id,
        source_path=str(info.path),
        title=info.title,
        author=info.author,
        page_count=info.page_count,
    )

    document = DocumentResult(metadata=metadata, pages=pages)
    document.chunks = chunk_document(document)

    elapsed = time.perf_counter() - start
    return PipelineRun(
        document=document,
        elapsed_seconds=elapsed,
        ocr_seconds=ocr_total,
        page_seconds=page_seconds,
        repeated_texts=repeated,
    )
