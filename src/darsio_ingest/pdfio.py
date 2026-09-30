"""PDF input/output: lazy rendering, metadata, checksums.

Pages are rendered lazily (one at a time) so 100+ page documents never hold
more than a single page image in memory. The same reader serves both ingest
paths: the text path reads the content stream objects, the OCR path renders
page images.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pymupdf  # PyMuPDF
from PIL import Image


@dataclass
class PageInput:
    page_number: int  # 1-based
    image: Image.Image
    width: float  # page size in PDF points
    height: float


@dataclass
class DocumentInfo:
    path: Path
    document_id: str
    page_count: int
    title: str | None
    author: str | None
    checksum_sha256: str = ""


def checksum_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_document_id(path: Path, checksum: str | None = None) -> str:
    """Stable id: filename stem + short content hash, e.g. `olom-a1b2c3d4`."""
    digest = (checksum or checksum_sha256(path))[:8]
    return f"{path.stem}-{digest}"


def open_document(path: str | Path) -> DocumentInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    checksum = checksum_sha256(path)
    doc = pymupdf.open(path)
    try:
        meta = doc.metadata or {}
        return DocumentInfo(
            path=path,
            document_id=compute_document_id(path, checksum),
            page_count=doc.page_count,
            title=meta.get("title") or None,
            author=meta.get("author") or None,
            checksum_sha256=checksum,
        )
    finally:
        doc.close()


def _render(doc: pymupdf.Document, index: int, dpi: int) -> PageInput:
    page = doc[index]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return PageInput(
        page_number=index + 1,
        image=image,
        width=page.rect.width,
        height=page.rect.height,
    )


def normalize_pages(
    pages: "list[int] | range | None", page_count: int
) -> list[int] | None:
    """Normalize a caller-supplied page selection to a sorted list.

    Accepts list[int], a range, or None (all pages). Out-of-range pages are
    silently dropped; empty selection means "all pages".
    """
    if pages is None:
        return None
    if isinstance(pages, range):
        pages = list(pages)
    selected = sorted({n for n in pages if 1 <= n <= page_count})
    return selected or None


def iter_pages(
    path: str | Path,
    dpi: int = 200,
    page_numbers: "list[int] | range | None" = None,
) -> Iterator[PageInput]:
    """Yield rendered pages one by one (1-based page numbers).

    `page_numbers` selects a subset (list or range); None means all pages.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = pymupdf.open(path)
    try:
        indices = (
            [n - 1 for n in normalize_pages(page_numbers, doc.page_count)]
            if page_numbers is not None
            else range(doc.page_count)
        )
        for index in indices:
            yield _render(doc, index, dpi)
    finally:
        doc.close()


def load_document(
    path: str | Path, dpi: int = 200
) -> tuple[DocumentInfo, list[PageInput]]:
    """Render all pages up-front. Only for small docs / tests -- prefer iter_pages."""
    info = open_document(path)
    pages = list(iter_pages(path, dpi=dpi))
    return info, pages


def save_page_image(image: Image.Image, out_dir: Path, page_number: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page_{page_number}.png"
    image.save(out_path)
    return out_path
