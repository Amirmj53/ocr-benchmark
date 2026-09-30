"""Shared data models for the Darsio ingestion engine.

Every artifact keeps its geometry and provenance so downstream systems
(embeddings, RAG, citations) can always trace a piece of text back to its
exact place on a page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LineType(str, Enum):
    BODY = "body"
    HEADING = "heading"
    SPECIAL = "special"
    NOISE = "noise"


class RegionRole(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    BODY = "body"
    SPECIAL = "special"
    NOISE = "noise"


class ChunkType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    SPECIAL = "special"
    TABLE = "table"


@dataclass
class OCRBlock:
    """One raw OCR detection box."""

    text: str
    score: float
    page_number: int = 0
    left: float = 0.0
    top: float = 0.0
    right: float = 0.0
    bottom: float = 0.0

    @property
    def width(self) -> float:
        return max(self.right - self.left, 0.0)

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 0.0)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "page_number": self.page_number,
            "bbox": [
                round(self.left, 1),
                round(self.top, 1),
                round(self.right, 1),
                round(self.bottom, 1),
            ],
        }


@dataclass
class OCRLine:
    """One visual text line assembled from one or more OCR blocks."""

    blocks: list[OCRBlock]
    page_number: int = 0

    @property
    def left(self) -> float:
        return min(b.left for b in self.blocks)

    @property
    def right(self) -> float:
        return max(b.right for b in self.blocks)

    @property
    def top(self) -> float:
        return min(b.top for b in self.blocks)

    @property
    def bottom(self) -> float:
        return max(b.bottom for b in self.blocks)

    @property
    def width(self) -> float:
        return max(self.right - self.left, 0.0)

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 0.0)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def text(self) -> str:
        """RTL line text: blocks ordered right-to-left, joined with spaces.

        Persian strings themselves are never reversed -- only the order of
        the blocks (visual order) is mapped to logical reading order.
        """
        return " ".join(b.text for b in self.blocks).strip()

    @property
    def min_score(self) -> float:
        return min(b.score for b in self.blocks)

    @property
    def mean_score(self) -> float:
        return sum(b.score for b in self.blocks) / len(self.blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "page_number": self.page_number,
            "bbox": [
                round(self.left, 1),
                round(self.top, 1),
                round(self.right, 1),
                round(self.bottom, 1),
            ],
            "score": round(self.mean_score, 4),
        }


@dataclass
class TextRegion:
    """A logical region: paragraph, heading, special block, noise strip."""

    role: RegionRole
    lines: list[OCRLine]
    page_number: int = 0
    column: int = 0
    heading_text: str | None = None  # nearest preceding heading, if any
    source_method: str = "ocr"  # "ocr" | "text" -- provenance for chunking

    @property
    def left(self) -> float:
        return min(line.left for line in self.lines)

    @property
    def right(self) -> float:
        return max(line.right for line in self.lines)

    @property
    def top(self) -> float:
        return min(line.top for line in self.lines)

    @property
    def bottom(self) -> float:
        return max(line.bottom for line in self.lines)

    @property
    def width(self) -> float:
        return max(self.right - self.left, 0.0)

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 0.0)

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines if line.text).strip()

    @property
    def mean_score(self) -> float:
        scores = [line.mean_score for line in self.lines]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "page_number": self.page_number,
            "column": self.column,
            "bbox": [
                round(self.left, 1),
                round(self.top, 1),
                round(self.right, 1),
                round(self.bottom, 1),
            ],
            "text": self.text,
            "score": round(self.mean_score, 4),
            "heading": self.heading_text,
        }


@dataclass
class PageResult:
    """Structured output for one processed PDF page (either path)."""

    page_number: int  # 1-based
    width: float = 0.0
    height: float = 0.0
    blocks: list[OCRBlock] = field(default_factory=list)
    lines: list[OCRLine] = field(default_factory=list)
    regions: list[TextRegion] = field(default_factory=list)
    column_count: int = 1
    headings: list[str] = field(default_factory=list)
    text: str = ""  # normalized full-page text (paragraphs separated by \n\n)
    source_method: str = "ocr"  # "ocr" | "text"
    status: str = ""  # PageStatus value from diagnose

    def to_dict(self, include_blocks: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "page_number": self.page_number,
            "size": [round(self.width, 1), round(self.height, 1)],
            "column_count": self.column_count,
            "headings": self.headings,
            "regions": [r.to_dict() for r in self.regions],
            "text": self.text,
            "source_method": self.source_method,
            "status": self.status,
        }
        if include_blocks:
            data["blocks"] = [b.to_dict() for b in self.blocks]
        return data


@dataclass
class DocumentMetadata:
    document_id: str
    source_path: str
    title: str | None = None
    author: str | None = None
    page_count: int = 0
    checksum_sha256: str = ""
    parser: str = "ocr"
    ocr_model: str = "rapidocr-ppocrv5-arabic-mobile"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "title": self.title,
            "author": self.author,
            "page_count": self.page_count,
            "checksum_sha256": self.checksum_sha256,
            "parser": self.parser,
            "ocr_model": self.ocr_model,
            **self.extra,
        }


@dataclass
class Chunk:
    """A retrieval-ready chunk with full provenance metadata (stable RAG schema)."""

    chunk_id: str
    document_id: str
    chunk_index: int
    chunk_type: ChunkType
    text: str
    page_start: int
    page_end: int
    section: str | None = None
    section_path: list[str] = field(default_factory=list)
    bbox: list[float] | None = None  # union bbox of chunk content (top-level pages)
    page_bboxes: dict[str, list[float]] | None = None  # {"3": [l,t,r,b]}
    score: float = 0.0  # mean OCR confidence (0.0-1.0 for the text path)
    language: str = "fa"
    char_count: int = 0
    # Provenance (v0.2 fields -- part of the Darsio contract):
    source_method: str = "ocr"  # "ocr" | "text"
    engine_version: str = "0.2.0"
    normalized: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type.value,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "section_path": self.section_path,
            "bbox": self.bbox,
            "page_bboxes": self.page_bboxes,
            "score": round(self.score, 4),
            "language": self.language,
            "char_count": self.char_count,
            "source_method": self.source_method,
            "engine_version": self.engine_version,
            "normalized": self.normalized,
        }


@dataclass
class DocumentResult:
    """Final structured output for a whole document."""

    metadata: DocumentMetadata
    pages: list[PageResult] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    method_per_page: dict[int, str] = field(default_factory=dict)

    def to_dict(self, include_blocks: bool = False) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "pages": [p.to_dict(include_blocks=include_blocks) for p in self.pages],
            "chunks": [c.to_dict() for c in self.chunks],
            "method_per_page": {str(k): v for k, v in self.method_per_page.items()},
        }
