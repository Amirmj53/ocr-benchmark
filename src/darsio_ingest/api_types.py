"""Stable public API types for the Darsio ingestion engine.

Everything in this module is part of the contract between this package and
the Darsio backend (app/services/ai/*). Changing these fields is a breaking
change; add new fields instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

# Bumped when processing output could change for the same input bytes
# (layout rules, normalization, chunking, OCR model). Part of the page-cache
# key so stale cached pages are never served after an engine upgrade.
ENGINE_VERSION = "0.2.0"

# Identifier of the OCR model combination (recorded, not discovered at
# runtime; the engine choice is fixed by documented benchmark).
OCR_ENGINE_ID = "rapidocr-ppocrv5-arabic-mobile"

# Full cache-facing engine identity.
ENGINE_ID = f"darsio-ingest-{ENGINE_VERSION}+{OCR_ENGINE_ID}"


class PageStatus(str, Enum):
    """Diagnosis of one PDF page (cheap, object-level, no OCR)."""

    TEXT_OK = "text_ok"
    TEXT_PLUS_VISUAL = "text_plus_visual"
    TEXT_BROKEN = "text_broken"
    SCANNED = "scanned"
    EMPTY = "empty"


class RecommendedMethod(str, Enum):
    """What the engine recommends per page. Diagnose never picks ranges."""

    TEXT_EXTRACT_SORTED = "text_extract_sorted"
    OCR = "ocr"
    HYBRID = "hybrid"
    SKIP = "skip"  # empty pages: nothing to extract


# Caller-facing processing mode (IngestJob.mode / ProcessOptions.mode).
IngestMode = Literal["auto", "text_only", "ocr_only"]

# How a page was actually turned into text.
SourceMethod = Literal["text", "ocr"]


@dataclass
class PageDiagnosis:
    """Diagnosis for a single page."""

    page_number: int  # 1-based
    status: PageStatus
    recommended_method: RecommendedMethod
    # Raw signals (useful for debugging and for quota decisions upstream).
    char_count: int = 0
    readable_char_count: int = 0  # excludes PUA / replacement / control chars
    persian_char_count: int = 0
    pua_char_count: int = 0
    replacement_char_count: int = 0
    control_char_count: int = 0
    text_block_count: int = 0
    image_count: int = 0
    image_area_ratio: float = 0.0  # sum(image bbox areas) / page area
    drawing_count: int = 0
    font_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "status": self.status.value,
            "recommended_method": self.recommended_method.value,
            "char_count": self.char_count,
            "readable_char_count": self.readable_char_count,
            "persian_char_count": self.persian_char_count,
            "pua_char_count": self.pua_char_count,
            "replacement_char_count": self.replacement_char_count,
            "control_char_count": self.control_char_count,
            "text_block_count": self.text_block_count,
            "image_count": self.image_count,
            "image_area_ratio": round(self.image_area_ratio, 4),
            "drawing_count": self.drawing_count,
            "font_count": self.font_count,
        }


@dataclass
class DocumentDiagnosis:
    """Aggregate diagnosis for a document plus the per-page map."""

    source_path: str
    document_id: str
    checksum_sha256: str
    page_count: int
    pages: dict[int, PageDiagnosis] = field(default_factory=dict)
    status_counts: dict[str, int] = field(default_factory=dict)
    dominant_status: PageStatus = PageStatus.EMPTY
    recommended_method: RecommendedMethod = RecommendedMethod.SKIP

    def pages_with_status(self, *statuses: PageStatus) -> list[int]:
        return sorted(
            p.page_number for p in self.pages.values() if p.status in statuses
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "document_id": self.document_id,
            "checksum_sha256": self.checksum_sha256,
            "page_count": self.page_count,
            "status_counts": self.status_counts,
            "dominant_status": self.dominant_status.value,
            "recommended_method": self.recommended_method.value,
            "pages": {
                str(n): d.to_dict() for n, d in sorted(self.pages.items())
            },
        }


@dataclass
class ProcessOptions:
    """Options for process_pages / process_document / run_ingest_job."""

    mode: IngestMode = "auto"
    dpi: int = 200
    # Chunking policy: False keeps chunks within one page (pamphlets,
    # slides); True lets paragraphs flow across page breaks (textbooks).
    merge_across_pages: bool = False
    reset_sections_per_page: bool = True
    # Cross-page header/footer/watermark removal (needs >= 4 pages).
    strip_repeated: bool = True
    # Auto mode: if the text path yields fewer readable chars than this,
    # the page falls back to OCR (guards against silent empty extraction).
    min_text_chars: int = 40
    # Hybrid pages (text_plus_visual): OCR them too when True.
    ocr_hybrid_pages: bool = False
    # Optional image preprocessing (A/B tested: OFF by default, slower AND
    # worse on the benchmark document).
    grayscale: bool = False
    contrast: float = 1.0
    sharpen: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "dpi": self.dpi,
            "merge_across_pages": self.merge_across_pages,
            "reset_sections_per_page": self.reset_sections_per_page,
            "strip_repeated": self.strip_repeated,
            "min_text_chars": self.min_text_chars,
            "ocr_hybrid_pages": self.ocr_hybrid_pages,
            "grayscale": self.grayscale,
            "contrast": self.contrast,
            "sharpen": self.sharpen,
        }


@dataclass
class PageIngestResult:
    """Result of ingesting one page (spec: per-page result)."""

    page_number: int
    text: str
    status: PageStatus
    source_method: SourceMethod
    cache_hit: bool = False
    fallback_used: bool = False  # text path fell back to OCR
    chunks: list[Any] = field(default_factory=list)  # list[Chunk]
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_chunks: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "page_number": self.page_number,
            "status": self.status.value,
            "source_method": self.source_method,
            "cache_hit": self.cache_hit,
            "fallback_used": self.fallback_used,
            "char_count": len(self.text),
            "chunk_count": len(self.chunks),
            "timings": {k: round(v, 4) for k, v in self.timings.items()},
        }
        if include_chunks:
            data["chunks"] = [c.to_dict() for c in self.chunks]
        return data


@dataclass
class IngestJob:
    """A unit of ingest work. Darsio can serialize this into a DB queue."""

    document_id: str
    pdf_path: str
    pages: list[int]
    mode: IngestMode = "auto"
    dpi: int = 200
    merge_across_pages: bool = False
    # Optional checksum of the source PDF; recomputed when empty.
    checksum_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "pdf_path": self.pdf_path,
            "pages": self.pages,
            "mode": self.mode,
            "dpi": self.dpi,
            "merge_across_pages": self.merge_across_pages,
            "checksum_sha256": self.checksum_sha256,
        }


@dataclass
class IngestJobResult:
    """Result of running one IngestJob."""

    job: IngestJob
    pages_done: list[int] = field(default_factory=list)
    chunks: list[Any] = field(default_factory=list)  # list[Chunk]
    timings: dict[str, Any] = field(default_factory=dict)
    method_per_page: dict[int, str] = field(default_factory=dict)
    status_per_page: dict[int, str] = field(default_factory=dict)
    document: Any = None  # DocumentResult, when the full doc was requested

    def to_dict(self, include_chunks: bool = False) -> dict[str, Any]:
        return {
            "job": self.job.to_dict(),
            "pages_done": self.pages_done,
            "method_per_page": {str(k): v for k, v in self.method_per_page.items()},
            "status_per_page": {str(k): v for k, v in self.status_per_page.items()},
            "timings": self.timings,
            "chunk_count": len(self.chunks),
            "chunks": [c.to_dict() for c in self.chunks] if include_chunks else None,
        }


@dataclass
class CachedPage:
    """One row of the page cache. Darsio maps this onto its DB directly."""

    document_id: str
    page_no: int
    text: str
    # Identity of how this page was produced. A cached page is a hit only
    # when ALL of these match the current request.
    checksum_sha256: str = ""
    dpi: int = 200
    source_method: SourceMethod = "ocr"
    engine_version: str = ENGINE_VERSION
    created_at: str = ""

    def matches(self, checksum_sha256: str, dpi: int, source_method: str) -> bool:
        return (
            self.engine_version == ENGINE_VERSION
            and self.checksum_sha256 == checksum_sha256
            and self.dpi == dpi
            and self.source_method == source_method
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "page_no": self.page_no,
            "text": self.text,
            "checksum_sha256": self.checksum_sha256,
            "dpi": self.dpi,
            "source_method": self.source_method,
            "engine_version": self.engine_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CachedPage:
        return cls(
            document_id=data["document_id"],
            page_no=int(data["page_no"]),
            text=data.get("text", ""),
            checksum_sha256=data.get("checksum_sha256", ""),
            dpi=int(data.get("dpi", 200)),
            source_method=data.get("source_method", "ocr"),
            engine_version=data.get("engine_version", ""),
            created_at=data.get("created_at", ""),
        )
