"""Diagnose layer: what kind of page/document is this, without doing OCR.

Answers the product question cheaply, at the pymupdf object level:

* ``text_ok``           healthy extractable Unicode text (Persian or Latin)
* ``text_plus_visual``  healthy text AND significant image/formula graphics
* ``text_broken``       text objects exist but encoding is damaged (PUA /
                        replacement spam) -- treat as an OCR page
* ``scanned``           little/no text, image-heavy
* ``empty``             nothing extractable, nothing visible

Design rule: diagnose NEVER decides page ranges or spends time on OCR. It
reports the page type and the recommended method
(text_extract_sorted | ocr | hybrid | skip); the product/quota layer in
Darsio decides what to do with that information.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from darsio_ingest.api_types import (
    DocumentDiagnosis,
    PageDiagnosis,
    PageStatus,
    RecommendedMethod,
)
from darsio_ingest.pdfio import checksum_sha256, compute_document_id

# ---------------------------------------------------------------------------
# Thresholds (single source of truth; tests pin these behaviors)
# ---------------------------------------------------------------------------

# Page counts as "has a text layer" above this many chars of extractable text.
TEXT_CHARS_MIN = 50
# Below this, a present-but-tiny text layer is treated as absent.
BROKEN_TEXT_CHARS_MIN = 20
# Fraction of readable text chars that must be Persian/Arabic script for the
# text layer to count as healthy Persian content.
PERSIAN_RATIO_OK = 0.20
# PUA + replacement chars above this fraction of all text chars => broken.
BROKEN_CHAR_RATIO = 0.10
# Images cover at least this fraction of the page area => image-heavy.
IMAGE_AREA_RATIO = 0.30
# At least this many placed images with meaningful coverage => image-heavy.
IMAGE_COUNT_MIN = 1
# This many or more vector drawings => visually rich even without rasters.
DRAWING_COUNT_VISUAL = 20


@dataclass
class PageTextStats:
    """Raw object-level signals for one page. Pure data, trivially testable."""

    char_count: int = 0
    readable_char_count: int = 0  # excludes PUA/replacement/control chars
    persian_char_count: int = 0
    pua_char_count: int = 0
    replacement_char_count: int = 0
    control_char_count: int = 0  # NUL/C0/DEL spam (not tab/newline/CR)
    text_block_count: int = 0
    image_count: int = 0
    image_area_ratio: float = 0.0
    drawing_count: int = 0
    font_count: int = 0


# ---------------------------------------------------------------------------
# Stats extraction (the only pymupdf-touching part)
# ---------------------------------------------------------------------------

_PUA_START, _PUA_END = 0xE000, 0xF8FF
_READABLE_PAD = {
    "\n", "\r", "\t", " ", "\u200c",  # ZWNJ is meaningful Persian
}


def is_persian_char(ch: str) -> bool:
    """Persian/Arabic script, including presentation forms.

    Many Persian PDF generators (including pymupdf's own HTML writer) store
    text as Arabic presentation forms (FB50-FDFF); those must count as
    Persian, not as broken or Latin.
    """
    code = ord(ch)
    return (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0xFB50 <= code <= 0xFDFF
    )


def is_broken_char(ch: str) -> bool:
    """Encoding-damage markers: PUA, replacement char, NUL/control spam."""
    code = ord(ch)
    if _PUA_START <= code <= _PUA_END:
        return True
    if ch == "\ufffd":  # REPLACEMENT CHARACTER
        return True
    # C0 controls except tab/newline/carriage-return, plus DEL. Broken
    # generators emit runs of NULs where glyphs should be.
    if ch in "\n\r\t":
        return False
    return code < 0x20 or code == 0x7F


def compute_page_text_stats(page: "pymupdf.Page") -> PageTextStats:
    """Collect object-level text/graphics signals for one page."""
    stats = PageTextStats()

    raw_text = page.get_text("text") or ""
    stats.char_count = len(raw_text)

    for ch in raw_text:
        if is_broken_char(ch):
            if _PUA_START <= ord(ch) <= _PUA_END:
                stats.pua_char_count += 1
            elif ch == "\ufffd":
                stats.replacement_char_count += 1
            else:
                stats.control_char_count += 1
        elif ch.isalnum() or ch in _READABLE_PAD:
            stats.readable_char_count += 1

    stats.persian_char_count = sum(
        1 for ch in raw_text if is_persian_char(ch)
    )

    blocks = page.get_text("blocks") or []
    stats.text_block_count = sum(
        1 for b in blocks if b and (b[4] or "").strip()
    )

    fonts = page.get_fonts(full=True) or []
    stats.font_count = len(fonts)

    images = page.get_images(full=True) or []
    stats.image_count = len(images)

    # Image coverage: sum of placed image bbox areas / page area.
    page_area = abs(page.rect) if page.rect else 0.0
    covered = 0.0
    if page_area > 0:
        for info in images:
            xref = info[0]
            try:
                for rect in page.get_image_rects(xref):
                    covered += abs(rect)
            except Exception:
                continue
    stats.image_area_ratio = min(covered / page_area, 1.0) if page_area else 0.0

    stats.drawing_count = len(page.get_drawings() or [])

    return stats


# ---------------------------------------------------------------------------
# Classification (pure functions over stats)
# ---------------------------------------------------------------------------

def broken_ratio(stats: PageTextStats) -> float:
    """Broken chars (PUA + replacement + control spam) / all extracted chars."""
    if stats.char_count == 0:
        return 0.0
    return (
        stats.pua_char_count
        + stats.replacement_char_count
        + stats.control_char_count
    ) / stats.char_count


def persian_ratio(stats: PageTextStats) -> float:
    """Persian-script chars as a fraction of readable chars."""
    if stats.readable_char_count == 0:
        return 0.0
    return stats.persian_char_count / stats.readable_char_count


def is_visual(stats: PageTextStats) -> bool:
    """Significant image or vector-graphic presence."""
    return (
        stats.image_count >= IMAGE_COUNT_MIN
        and stats.image_area_ratio >= IMAGE_AREA_RATIO
    ) or stats.drawing_count >= DRAWING_COUNT_VISUAL


def classify_page(stats: PageTextStats) -> PageStatus:
    """Classify one page from its stats. Pure; no I/O."""
    broken = broken_ratio(stats)

    if stats.char_count == 0 and stats.image_count == 0 and stats.drawing_count == 0:
        return PageStatus.EMPTY

    if stats.char_count < TEXT_CHARS_MIN:
        # Tiny text layer. Broken if it is mostly PUA junk, scanned otherwise
        # (the visible content lives in the raster).
        if stats.char_count > 0 and broken >= BROKEN_CHAR_RATIO:
            return PageStatus.TEXT_BROKEN
        if stats.image_count == 0 and stats.drawing_count == 0:
            return PageStatus.EMPTY if stats.char_count == 0 else PageStatus.SCANNED
        return PageStatus.SCANNED

    if broken >= BROKEN_CHAR_RATIO:
        return PageStatus.TEXT_BROKEN

    # Healthy text layer. A text layer with almost no letters at all (only
    # digits/punctuation, no Persian, no Latin) is a typographic ghost.
    letters = stats.persian_char_count + _latin_letter_count_estimate(stats)
    if letters < BROKEN_TEXT_CHARS_MIN:
        return PageStatus.TEXT_BROKEN

    if is_visual(stats):
        return PageStatus.TEXT_PLUS_VISUAL
    return PageStatus.TEXT_OK


def _latin_letter_count_estimate(stats: PageTextStats) -> int:
    # readable includes digits and separators; Persian count is exact. Treat
    # the remainder generously: classify_page only needs a floor.
    return max(stats.readable_char_count - stats.persian_char_count, 0)


def recommend_method(status: PageStatus) -> RecommendedMethod:
    """Recommended ingest method for a page status. Pure."""
    if status == PageStatus.TEXT_OK:
        return RecommendedMethod.TEXT_EXTRACT_SORTED
    if status == PageStatus.TEXT_PLUS_VISUAL:
        return RecommendedMethod.HYBRID
    if status == PageStatus.TEXT_BROKEN:
        return RecommendedMethod.OCR
    if status == PageStatus.SCANNED:
        return RecommendedMethod.OCR
    return RecommendedMethod.SKIP


def make_page_diagnosis(page_number: int, stats: PageTextStats) -> PageDiagnosis:
    status = classify_page(stats)
    return PageDiagnosis(
        page_number=page_number,
        status=status,
        recommended_method=recommend_method(status),
        char_count=stats.char_count,
        readable_char_count=stats.readable_char_count,
        persian_char_count=stats.persian_char_count,
        pua_char_count=stats.pua_char_count,
        replacement_char_count=stats.replacement_char_count,
        control_char_count=stats.control_char_count,
        text_block_count=stats.text_block_count,
        image_count=stats.image_count,
        image_area_ratio=stats.image_area_ratio,
        drawing_count=stats.drawing_count,
        font_count=stats.font_count,
    )


# ---------------------------------------------------------------------------
# Aggregate diagnosis
# ---------------------------------------------------------------------------

_STATUS_ORDER: list[PageStatus] = [
    PageStatus.TEXT_OK,
    PageStatus.TEXT_PLUS_VISUAL,
    PageStatus.TEXT_BROKEN,
    PageStatus.SCANNED,
    PageStatus.EMPTY,
]


def diagnose_document(
    path: str | Path,
    pages: "list[int] | range | None" = None,
) -> DocumentDiagnosis:
    """Diagnose a PDF (all or selected pages). Cheap: no rendering, no OCR."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    checksum = checksum_sha256(path)
    doc = pymupdf.open(path)
    try:
        if pages is None:
            selected = list(range(1, doc.page_count + 1))
        else:
            selected = sorted(
                {n for n in pages if isinstance(n, int) and 1 <= n <= doc.page_count}
            )

        page_diagnoses: dict[int, PageDiagnosis] = {}
        for number in selected:
            page = doc[number - 1]
            stats = compute_page_text_stats(page)
            page_diagnoses[number] = make_page_diagnosis(number, stats)
    finally:
        doc.close()

    counts = {status.value: 0 for status in _STATUS_ORDER}
    for diagnosis in page_diagnoses.values():
        counts[diagnosis.status.value] += 1

    dominant = PageStatus.EMPTY
    if counts:
        dominant_value = max(counts.items(), key=lambda kv: kv[1])[0]
        dominant = PageStatus(dominant_value)

    # Document-level recommendation: any OCR-needing content dominates the
    # method choice conservatively -- a mixed document is hybrid if it has
    # both healthy text and OCR-needing pages, OCR if it is mostly scanned.
    methods = {d.recommended_method for d in page_diagnoses.values()}
    has_ocr = RecommendedMethod.OCR in methods
    has_text = (
        RecommendedMethod.TEXT_EXTRACT_SORTED in methods
        or RecommendedMethod.HYBRID in methods
    )
    if has_ocr and has_text:
        doc_method = RecommendedMethod.HYBRID
    elif has_ocr:
        doc_method = RecommendedMethod.OCR
    elif has_text:
        doc_method = RecommendedMethod.TEXT_EXTRACT_SORTED
    else:
        doc_method = RecommendedMethod.SKIP

    return DocumentDiagnosis(
        source_path=str(path),
        document_id=compute_document_id(path, checksum),
        checksum_sha256=checksum,
        page_count=len(selected) if pages is not None else _page_count(path),
        pages=page_diagnoses,
        status_counts=counts,
        dominant_status=dominant,
        recommended_method=doc_method,
    )


def _page_count(path: Path) -> int:
    doc = pymupdf.open(path)
    try:
        return doc.page_count
    finally:
        doc.close()


def diagnose_pdf(
    path: str | Path,
    pages: "list[int] | range | None" = None,
) -> DocumentDiagnosis:
    """Public alias used by Darsio: diagnose_pdf(pdf)."""
    return diagnose_document(path, pages=pages)
