"""Dual-path ingestion pipeline: text extraction OR OCR, per page.

Decision flow per page (mode="auto"):

    diagnose page (object-level, cheap)
      text_ok          -> extract text layer, NO OCR
      text_plus_visual -> extract text; OCR too only if options allow
      text_broken      -> render + OCR
      scanned          -> render + OCR
      empty            -> skip (cached as empty)

The caller (Darsio quota/product layer) decides WHICH pages to process;
this module only decides HOW, per page. All public entry points accept
page selections as list[int] or range.

Performance contract: the OCR engine is a process-wide singleton (models
load once); pages are rendered lazily; memory holds one page image at a
time (~8s/page OCR, ~1GB peak on the reference machine).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from darsio_ingest.api_types import (
    ENGINE_VERSION,
    CachedPage,
    DocumentDiagnosis,
    IngestJob,
    IngestJobResult,
    PageIngestResult,
    ProcessOptions,
)
from darsio_ingest.cache import PageCache
from darsio_ingest.chunking import chunk_regions
from darsio_ingest.diagnose import diagnose_document
from darsio_ingest.extract import build_text_page_results
from darsio_ingest.layout import build_page_result, strip_repeated_content
from darsio_ingest.models import Chunk, DocumentMetadata, DocumentResult, PageResult
from darsio_ingest.normalize import normalize_text
from darsio_ingest.ocr import get_ocr_engine
from darsio_ingest.ocr.preprocessing import preprocess_image
from darsio_ingest.pdfio import DocumentInfo, iter_pages, normalize_pages, open_document


@dataclass
class PipelineConfig:
    """Kept for backwards compatibility with the benchmark scripts."""

    dpi: int = 200
    grayscale: bool = False
    contrast: float = 1.0
    sharpen: bool = False
    strip_repeated: bool = True
    keep_noise_blocks: bool = False
    save_debug_images: bool = False
    debug_image_dir: Path = Path("data/output/debug")

    def to_options(self) -> ProcessOptions:
        return ProcessOptions(
            dpi=self.dpi,
            strip_repeated=self.strip_repeated,
            grayscale=self.grayscale,
            contrast=self.contrast,
            sharpen=self.sharpen,
        )


@dataclass
class PipelineRun:
    """Full-document run output (legacy shape, still used by scripts)."""

    document: DocumentResult
    elapsed_seconds: float
    ocr_seconds: float
    page_seconds: dict[int, float] = field(default_factory=dict)
    repeated_texts: set[str] = field(default_factory=set)
    profiling: dict = field(default_factory=dict)
    diagnosis: DocumentDiagnosis | None = None


def _page_to_ingest_result(
    page: PageResult,
    *,
    cache_hit: bool = False,
    fallback_used: bool = False,
    chunks: list[Chunk] | None = None,
    timings: dict[str, float] | None = None,
) -> PageIngestResult:
    return PageIngestResult(
        page_number=page.page_number,
        text=page.text,
        status=page.status,
        source_method=page.source_method,  # type: ignore[arg-type]
        cache_hit=cache_hit,
        fallback_used=fallback_used,
        chunks=chunks or [],
        timings=timings or {},
    )


def process_page_ocr(
    page_input,
    options: ProcessOptions,
    status: str = "",
) -> tuple[PageResult, dict[str, float]]:
    """OCR path for one rendered page: preprocess -> OCR -> layout -> normalize."""
    engine = get_ocr_engine()

    timings: dict[str, float] = {}

    start = time.perf_counter()
    image = preprocess_image(
        page_input.image,
        grayscale=options.grayscale,
        contrast=options.contrast,
        sharpen=options.sharpen,
    )
    timings["preprocess"] = time.perf_counter() - start

    blocks, ocr_seconds = engine.run_page(image, page_input.page_number)
    timings["ocr"] = ocr_seconds

    start = time.perf_counter()
    # Image pixels == PDF points * dpi/72; layout works in pixel space
    # consistently, so pass the pixel size as the page size.
    page = build_page_result(
        page_number=page_input.page_number,
        width=float(page_input.image.size[0]),
        height=float(page_input.image.size[1]),
        blocks=blocks,
        source_method="ocr",
        status=status,
    )
    timings["layout"] = time.perf_counter() - start

    # Normalize text while geometry is preserved.
    start = time.perf_counter()
    for block in page.blocks:
        block.text = normalize_text(block.text)
    for line in page.lines:
        for block in line.blocks:
            block.text = normalize_text(block.text)
    for region in page.regions:
        for line in region.lines:
            for block in line.blocks:
                block.text = normalize_text(block.text)
        if region.heading_text:
            region.heading_text = normalize_text(region.heading_text)
    page.headings = [normalize_text(h) for h in page.headings]
    page.text = normalize_text(page.text)
    timings["normalize"] = time.perf_counter() - start

    return page, timings


def _chunks_for_page(
    document_id: str, page: PageResult, options: ProcessOptions
) -> list[Chunk]:
    return chunk_regions(
        document_id,
        page.regions,
        merge_across_pages=False,  # per-page chunking never spans pages
        reset_sections_per_page=True,
        default_source_method=page.source_method,
    )


def process_pages(
    pdf_path: str | Path,
    pages: "list[int] | range | None" = None,
    options: ProcessOptions | None = None,
    cache: PageCache | None = None,
    document_id: str | None = None,
    checksum_sha256: str = "",
) -> list[PageIngestResult]:
    """Process selected pages with the dual-path engine. Spec entry point.

    Idempotent when a cache is given: a page whose cached identity matches
    (checksum + dpi + method + engine version) is skipped entirely.

    `document_id` defaults to the content-derived id from pdfio.
    """
    options = options or ProcessOptions()
    info = open_document(pdf_path)
    document_id = document_id or info.document_id
    checksum = checksum_sha256 or info.checksum_sha256
    selected = normalize_pages(pages, info.page_count)

    diagnosis = diagnose_document(pdf_path, pages=selected)
    results = _process_selected_pages(
        pdf_path,
        document_id,
        checksum,
        diagnosis,
        selected,
        options,
        cache,
    )
    return results


def _process_selected_pages(
    pdf_path: str | Path,
    document_id: str,
    checksum: str,
    diagnosis: DocumentDiagnosis,
    selected: list[int] | None,
    options: ProcessOptions,
    cache: PageCache | None,
) -> list[PageIngestResult]:
    """Shared per-page driver for process_pages and process_document."""
    page_numbers = selected or diagnosis.pages and sorted(diagnosis.pages) or []
    if not page_numbers:
        return []

    # Split by recommended method; cache hits drop out first.
    ocr_pages: list[int] = []
    text_pages: list[int] = []
    results: dict[int, PageIngestResult] = {}

    info = open_document(pdf_path)

    for number in page_numbers:
        page_diag = diagnosis.pages.get(number)
        if page_diag is None:
            continue

        status = page_diag.status.value
        method = page_diag.recommended_method

        # Cache check (identity-aware). Must use the same method string
        # this request will store, see _route_method().
        if cache is not None:
            cached = cache.get_cached_page(document_id, number)
            route = _route_method(status, method, options)
            if cached is not None and cached.matches(checksum, options.dpi, route):
                page_result = PageResult(
                    page_number=number,
                    text=cached.text,
                    source_method=cached.source_method,
                    status=status,
                )
                results[number] = _page_to_ingest_result(
                    page_result, cache_hit=True, timings={"cache_hit": 0.0}
                )
                continue

        if status == "empty" and options.mode != "ocr_only":
            results[number] = _page_to_ingest_result(
                PageResult(page_number=number, text="", status=status),
                timings={"empty": 0.0},
            )
            continue

        if options.mode == "text_only":
            text_pages.append(number)
        elif options.mode == "ocr_only":
            ocr_pages.append(number)
        else:  # auto
            if method == "ocr" or status == "text_broken" or status == "scanned":
                ocr_pages.append(number)
            elif method == "hybrid":
                if options.ocr_hybrid_pages:
                    ocr_pages.append(number)
                else:
                    text_pages.append(number)
            else:
                text_pages.append(number)

    # Text path: one document open for the whole batch.
    for page_result in build_text_page_results(
        pdf_path, text_pages, status_per_page={n: diagnosis.pages[n].status.value for n in text_pages}
    ):
        # Auto-mode guard: silently empty extraction falls back to OCR.
        if (
            options.mode == "auto"
            and len(page_result.text) < options.min_text_chars
            and diagnosis.pages.get(page_result.page_number) is not None
            and diagnosis.pages[page_result.page_number].status.value
            in ("text_ok", "text_plus_visual")
        ):
            ocr_pages.append(page_result.page_number)
            continue

        chunks = _chunks_for_page(document_id, page_result, options)
        results[page_result.page_number] = _page_to_ingest_result(
            page_result, chunks=chunks, timings={"text_extract": 0.0}
        )
        if cache is not None:
            cache.upsert_page_text(
                CachedPage(
                    document_id=document_id,
                    page_no=page_result.page_number,
                    text=page_result.text,
                    checksum_sha256=checksum,
                    dpi=options.dpi,
                    source_method="text",
                    engine_version=ENGINE_VERSION,
                )
            )

    # OCR path: lazy rendering, singleton engine.
    if ocr_pages:
        for page_input in iter_pages(pdf_path, dpi=options.dpi, page_numbers=ocr_pages):
            number = page_input.page_number
            page, timings = process_page_ocr(
                page_input, options, status=diagnosis.pages[number].status.value
            )
            chunks = _chunks_for_page(document_id, page, options)
            results[number] = _page_to_ingest_result(
                page, chunks=chunks, timings=timings
            )
            if cache is not None:
                cache.upsert_page_text(
                    CachedPage(
                        document_id=document_id,
                        page_no=number,
                        text=page.text,
                        checksum_sha256=checksum,
                        dpi=options.dpi,
                        source_method="ocr",
                        engine_version=ENGINE_VERSION,
                    )
                )

    return [results[n] for n in sorted(results)]


def _route_method(status: str, method, options: ProcessOptions) -> str:
    """Mirror of the auto-routing decision, as the stored cache method.

    Keeps cache identity consistent: a page cached as "text" must match a
    request that would also route it to the text path.
    """
    if status == "empty" and options.mode != "ocr_only":
        return "text"
    if options.mode == "text_only":
        return "text"
    if options.mode == "ocr_only":
        return "ocr"
    if method.value == "ocr" or status in ("text_broken", "scanned"):
        return "ocr"
    if method.value == "hybrid":
        return "ocr" if options.ocr_hybrid_pages else "text"
    return "text"


def run_ingest_job(
    job: IngestJob,
    cache: PageCache | None = None,
    options: ProcessOptions | None = None,
) -> IngestJobResult:
    """Run one IngestJob end-to-end. Spec entry point for the queue worker."""
    options = options or ProcessOptions(
        mode=job.mode,
        dpi=job.dpi,
        merge_across_pages=job.merge_across_pages,
    )
    start = time.perf_counter()

    results = process_pages(
        job.pdf_path,
        pages=job.pages,
        options=options,
        cache=cache,
        document_id=job.document_id,
        checksum_sha256=job.checksum_sha256,
    )

    chunks: list[Chunk] = []
    method_per_page: dict[int, str] = {}
    status_per_page: dict[int, str] = {}
    for result in results:
        method_per_page[result.page_number] = result.source_method
        status_per_page[result.page_number] = result.status
        chunks.extend(result.chunks)

    # Re-index chunks across the whole job (chunk_regions numbered per page).
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index
        chunk.chunk_id = f"{job.document_id}-c{index:04d}"

    return IngestJobResult(
        job=job,
        pages_done=[r.page_number for r in results],
        chunks=chunks,
        timings={"total_seconds": time.perf_counter() - start},
        method_per_page=method_per_page,
        status_per_page=status_per_page,
    )


def process_document(
    pdf_path: str | Path,
    pages: "list[int] | range | None" = None,
    options: ProcessOptions | None = None,
    cache: PageCache | None = None,
) -> DocumentResult:
    """Process a document (or a page range of it) into a DocumentResult.

    merge_across_pages in options controls whether chunks may span page
    boundaries (True for flowing textbooks, False for pamphlets).
    """
    options = options or ProcessOptions()
    info: DocumentInfo = open_document(pdf_path)
    selected = normalize_pages(pages, info.page_count)

    diagnosis = diagnose_document(pdf_path, pages=selected)
    page_results = _process_selected_pages(
        pdf_path,
        info.document_id,
        info.checksum_sha256,
        diagnosis,
        selected,
        options,
        cache,
    )

    # Rebuild lightweight PageResults for chunking (keep region structure).
    # process_pages returns ingest results; chunk data is already attached
    # per page, but for a DocumentResult we re-chunk across the page stream
    # to honor merge_across_pages and running sections.
    doc = _reconstruct_page_results(pdf_path, selected, options, diagnosis)
    if doc is None:
        doc = DocumentResult(metadata=DocumentMetadata(document_id=info.document_id, source_path=str(info.path)))

    metadata = DocumentMetadata(
        document_id=info.document_id,
        source_path=str(info.path),
        title=info.title,
        author=info.author,
        page_count=info.page_count,
        checksum_sha256=info.checksum_sha256,
    )
    doc.metadata = metadata

    all_regions = []
    for page in doc.pages:
        all_regions.extend(page.regions)
    doc.chunks = chunk_regions(
        info.document_id,
        all_regions,
        merge_across_pages=options.merge_across_pages,
        reset_sections_per_page=options.reset_sections_per_page,
    )
    doc.method_per_page = {p.page_number: p.source_method for p in doc.pages}
    return doc


def _reconstruct_page_results(
    pdf_path: str | Path,
    selected: list[int] | None,
    options: ProcessOptions,
    diagnosis: DocumentDiagnosis,
) -> DocumentResult | None:
    """Route pages and produce structured PageResults (no caching here)."""
    from darsio_ingest.extract import build_text_page_results as _btext

    page_numbers = selected or sorted(diagnosis.pages)
    if not page_numbers:
        return None

    ocr_pages: list[int] = []
    text_pages: list[int] = []

    for number in page_numbers:
        page_diag = diagnosis.pages.get(number)
        if page_diag is None:
            continue
        status = page_diag.status.value
        if status == "empty":
            continue
        if options.mode == "text_only":
            text_pages.append(number)
        elif options.mode == "ocr_only":
            ocr_pages.append(number)
        elif page_diag.recommended_method == "ocr" or status in ("text_broken", "scanned"):
            ocr_pages.append(number)
        elif page_diag.recommended_method == "hybrid":
            (ocr_pages if options.ocr_hybrid_pages else text_pages).append(number)
        else:
            text_pages.append(number)

    pages_out: list[PageResult] = []

    text_results = _btext(
        pdf_path, text_pages, status_per_page={n: diagnosis.pages[n].status.value for n in text_pages}
    )
    for page_result in text_results:
        if (
            options.mode == "auto"
            and len(page_result.text) < options.min_text_chars
            and diagnosis.pages.get(page_result.page_number) is not None
            and diagnosis.pages[page_result.page_number].status.value
            in ("text_ok", "text_plus_visual")
        ):
            ocr_pages.append(page_result.page_number)
            continue
        pages_out.append(page_result)

    if ocr_pages:
        for page_input in iter_pages(pdf_path, dpi=options.dpi, page_numbers=ocr_pages):
            page, _ = process_page_ocr(
                page_input, options, status=diagnosis.pages[page_input.page_number].status.value
            )
            pages_out.append(page)

    pages_out.sort(key=lambda p: p.page_number)

    info = open_document(pdf_path)
    return DocumentResult(
        metadata=DocumentMetadata(
            document_id=info.document_id,
            source_path=str(info.path),
            page_count=info.page_count,
        ),
        pages=pages_out,
    )


def run_pipeline(
    pdf_path: str | Path,
    config: PipelineConfig | None = None,
    page_numbers: "list[int] | range | None" = None,
) -> PipelineRun:
    """Legacy full-document entry point (scripts + CLI bench).

    Diagnoses, routes, processes, strips repeated content, chunks.
    """
    config = config or PipelineConfig()
    options = config.to_options()
    start = time.perf_counter()

    info = open_document(pdf_path)
    selected = normalize_pages(page_numbers, info.page_count)
    diagnosis = diagnose_document(pdf_path, pages=selected)

    doc = _reconstruct_page_results(pdf_path, selected, options, diagnosis)
    document = doc or DocumentResult(
        metadata=DocumentMetadata(
            document_id=info.document_id,
            source_path=str(info.path),
            page_count=info.page_count,
        )
    )

    document.metadata = DocumentMetadata(
        document_id=info.document_id,
        source_path=str(info.path),
        title=info.title,
        author=info.author,
        page_count=info.page_count,
        checksum_sha256=info.checksum_sha256,
    )

    repeated: set[str] = set()
    if options.strip_repeated and len(document.pages) >= 4:
        repeated = strip_repeated_content(document.pages)

    chunking_start = time.perf_counter()
    all_regions: list = []
    for page in document.pages:
        all_regions.extend(page.regions)
    document.chunks = chunk_regions(
        info.document_id,
        all_regions,
        merge_across_pages=options.merge_across_pages,
        reset_sections_per_page=options.reset_sections_per_page,
    )
    chunking_seconds = time.perf_counter() - chunking_start

    elapsed = time.perf_counter() - start
    ocr_total = sum(p.timings.get("ocr", 0.0) for p in document.pages)
    page_seconds = {
        p.page_number: p.timings.get("ocr", 0.0)
        for p in document.pages
        if "ocr" in p.timings
    }

    document.method_per_page = {p.page_number: p.source_method for p in document.pages}

    return PipelineRun(
        document=document,
        elapsed_seconds=elapsed,
        ocr_seconds=ocr_total,
        page_seconds=page_seconds,
        repeated_texts=repeated,
        profiling={
            "chunking_seconds": chunking_seconds,
            "elapsed_seconds": elapsed,
            "pages": len(document.pages),
            "chunks": len(document.chunks),
        },
        diagnosis=diagnosis,
    )
