"""Darsio ingest engine: public API.

The import surface Darsio relies on:

    from darsio_ingest import (
        diagnose_pdf,          # cheap object-level page/doc classification
        process_pages,         # dual-path ingest for selected pages
        process_document,      # full DocumentResult with chunks
        run_ingest_job,        # IngestJob -> IngestJobResult (queue-ready)
        build_messages,        # study-mode prompt builders
        pack_context,          # deterministic chunk packing
    )

Heavier imports (RapidOCR / onnxruntime) are deferred until the OCR path
actually runs, so diagnosing or text-processing never pays model load.
"""

from darsio_ingest.api_types import (
    CachedPage,
    DocumentDiagnosis,
    ENGINE_ID,
    ENGINE_VERSION,
    IngestJob,
    IngestJobResult,
    PageDiagnosis,
    PageIngestResult,
    PageStatus,
    ProcessOptions,
    RecommendedMethod,
)
from darsio_ingest.cache import InMemoryPageCache, PageCache
from darsio_ingest.chunking import chunk_document, chunk_regions
from darsio_ingest.diagnose import diagnose_document, diagnose_pdf
from darsio_ingest.layout import build_page_result
from darsio_ingest.models import (
    Chunk,
    ChunkType,
    DocumentMetadata,
    DocumentResult,
    OCRBlock,
    OCRLine,
    PageResult,
    TextRegion,
)
from darsio_ingest.normalize import (
    has_persian,
    is_mostly_persian,
    normalize_text,
    persian_letter_ratio,
)
from darsio_ingest.pdfio import DocumentInfo, open_document
from darsio_ingest.pipeline import (
    PipelineConfig,
    PipelineRun,
    process_document,
    process_pages,
    run_ingest_job,
    run_pipeline,
)
from darsio_ingest.prompts import (
    build_messages,
    pack_context,
)

__version__ = ENGINE_VERSION

__all__ = [
    # version / identity
    "ENGINE_VERSION",
    "ENGINE_ID",
    "__version__",
    # diagnose
    "diagnose_pdf",
    "diagnose_document",
    "PageDiagnosis",
    "DocumentDiagnosis",
    "PageStatus",
    "RecommendedMethod",
    # pipeline
    "process_pages",
    "process_document",
    "run_ingest_job",
    "run_pipeline",
    "PipelineConfig",
    "PipelineRun",
    "ProcessOptions",
    "PageIngestResult",
    "IngestJob",
    "IngestJobResult",
    # cache
    "PageCache",
    "InMemoryPageCache",
    "CachedPage",
    # models
    "Chunk",
    "ChunkType",
    "DocumentResult",
    "DocumentMetadata",
    "PageResult",
    "TextRegion",
    "OCRBlock",
    "OCRLine",
    # chunking
    "chunk_document",
    "chunk_regions",
    # layout
    "build_page_result",
    # normalize
    "normalize_text",
    "has_persian",
    "is_mostly_persian",
    "persian_letter_ratio",
    # pdf io
    "open_document",
    "DocumentInfo",
    # prompts
    "build_messages",
    "pack_context",
]
