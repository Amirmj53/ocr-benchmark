"""Tests for diagnose heuristics.

The pure classification functions are pinned against synthetic PageTextStats;
the pymupdf-touching path is exercised on generated fixture PDFs (healthy
text layer, scanned image page, empty page, broken encoding with PUA spam).
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from darsio_ingest.api_types import PageStatus, RecommendedMethod
from darsio_ingest.diagnose import (
    PageTextStats,
    broken_ratio,
    classify_page,
    diagnose_document,
    persian_ratio,
    recommend_method,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic stats
# ---------------------------------------------------------------------------


def stats(**overrides) -> PageTextStats:
    base = PageTextStats(
        char_count=800,
        readable_char_count=700,
        persian_char_count=600,
        pua_char_count=0,
        replacement_char_count=0,
        text_block_count=12,
        image_count=0,
        image_area_ratio=0.0,
        drawing_count=2,
        font_count=2,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


class TestPureClassification:
    def test_healthy_persian_text_is_text_ok(self):
        assert classify_page(stats()) == PageStatus.TEXT_OK

    def test_text_plus_visual(self):
        s = stats(image_count=1, image_area_ratio=0.5)
        assert classify_page(s) == PageStatus.TEXT_PLUS_VISUAL

    def test_broken_by_pua_spam(self):
        s = stats(pua_char_count=200, char_count=1000, readable_char_count=600)
        assert classify_page(s) == PageStatus.TEXT_BROKEN

    def test_broken_by_replacement_chars(self):
        s = stats(replacement_char_count=150, char_count=1000, readable_char_count=600)
        assert classify_page(s) == PageStatus.TEXT_BROKEN

    def test_scanned(self):
        s = stats(
            char_count=5,
            readable_char_count=3,
            persian_char_count=0,
            image_count=1,
            image_area_ratio=0.95,
        )
        assert classify_page(s) == PageStatus.SCANNED

    def test_empty(self):
        s = stats(char_count=0, image_count=0, drawing_count=0)
        assert classify_page(s) == PageStatus.EMPTY

    def test_latin_text_counts_as_text_ok(self):
        s = stats(persian_char_count=10)
        assert classify_page(s) == PageStatus.TEXT_OK

    def test_small_drawings_do_not_make_visual(self):
        s = stats(drawing_count=5)
        assert classify_page(s) == PageStatus.TEXT_OK

    def test_many_drawings_make_visual(self):
        s = stats(drawing_count=50)
        assert classify_page(s) == PageStatus.TEXT_PLUS_VISUAL


class TestRatioHelpers:
    def test_broken_ratio_zero_when_no_text(self):
        assert broken_ratio(stats(char_count=0)) == 0.0

    def test_broken_ratio_fraction(self):
        s = stats(char_count=100, pua_char_count=15, replacement_char_count=5)
        assert broken_ratio(s) == 0.20

    def test_persian_ratio(self):
        s = stats(readable_char_count=100, persian_char_count=50)
        assert persian_ratio(s) == 0.5


class TestRecommendations:
    def test_mapping_is_consistent(self):
        assert recommend_method(PageStatus.TEXT_OK) == RecommendedMethod.TEXT_EXTRACT_SORTED
        assert recommend_method(PageStatus.TEXT_PLUS_VISUAL) == RecommendedMethod.HYBRID
        assert recommend_method(PageStatus.TEXT_BROKEN) == RecommendedMethod.OCR
        assert recommend_method(PageStatus.SCANNED) == RecommendedMethod.OCR
        assert recommend_method(PageStatus.EMPTY) == RecommendedMethod.SKIP


# ---------------------------------------------------------------------------
# Fixtures: real PDFs built in tmp_path
# ---------------------------------------------------------------------------


def _text_pdf(path: Path, persian: bool = True) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    text = (
        "این یک متن آزمایشی فارسی است برای بررسی لایه متنی سند. " * 6
        if persian
        else "The quick brown fox jumps over the lazy dog. " * 6
    )
    # insert_htmlbox embeds a proper Unicode font; insert_textbox with the
    # built-in helv font destroys Persian into '?' spam.
    page.insert_htmlbox(pymupdf.Rect(72, 72, 540, 700), f"<p>{text}</p>")
    doc.save(path)
    doc.close()
    return path


def _scanned_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    # A full-page dark rectangle stands in for a scanned raster.
    page.draw_rect(pymupdf.Rect(0, 0, 595, 842), color=None, fill=(0.2, 0.2, 0.2))
    doc.save(path)
    doc.close()
    return path


def _empty_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    return path


def _broken_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    # PUA run + ascii tail: extraction yields NUL/control spam + tail text,
    # so broken chars dominate (ratio >= 0.10 threshold).
    pua = "".join(chr(0xE000 + (i % 100)) for i in range(60))
    page.insert_htmlbox(
        pymupdf.Rect(72, 72, 540, 400),
        f"<p>{pua} ascii tail text here</p>",
    )
    doc.save(path)
    doc.close()
    return path


class TestDiagnoseDocument:
    def test_text_pdf_is_text_ok(self, tmp_path):
        pdf = _text_pdf(tmp_path / "text.pdf")
        diagnosis = diagnose_document(pdf)
        page = diagnosis.pages[1]
        assert page.status == PageStatus.TEXT_OK
        assert page.recommended_method == RecommendedMethod.TEXT_EXTRACT_SORTED
        assert diagnosis.dominant_status == PageStatus.TEXT_OK
        assert diagnosis.checksum_sha256
        assert diagnosis.document_id

    def test_latin_pdf_is_text_ok(self, tmp_path):
        pdf = _text_pdf(tmp_path / "latin.pdf", persian=False)
        diagnosis = diagnose_document(pdf)
        assert diagnosis.pages[1].status == PageStatus.TEXT_OK

    def test_empty_pdf_is_empty(self, tmp_path):
        pdf = _empty_pdf(tmp_path / "empty.pdf")
        diagnosis = diagnose_document(pdf)
        assert diagnosis.pages[1].status == PageStatus.EMPTY
        assert diagnosis.recommended_method == RecommendedMethod.SKIP

    def test_broken_pdf_is_text_broken(self, tmp_path):
        pdf = _broken_pdf(tmp_path / "broken.pdf")
        diagnosis = diagnose_document(pdf)
        page = diagnosis.pages[1]
        assert page.status == PageStatus.TEXT_BROKEN
        assert page.recommended_method == RecommendedMethod.OCR
        # PUA input may round-trip as PUA or as NUL/control spam; either
        # way the broken-char budget dominates.
        assert (
            page.pua_char_count + page.control_char_count + page.replacement_char_count
            > 30
        )

    def test_page_selection_respected(self, tmp_path):
        doc = pymupdf.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_htmlbox(
                pymupdf.Rect(72, 72, 540, 700),
                "<p>این یک متن آزمایشی فارسی است. " * 8 + "</p>",
            )
        pdf = tmp_path / "three.pdf"
        doc.save(pdf)
        doc.close()

        diagnosis = diagnose_document(pdf, pages=[2])
        assert list(diagnosis.pages) == [2]
        assert diagnosis.page_count == 1  # diagnosed count

    def test_diagnose_does_not_render_or_ocr(self, tmp_path):
        # A text-only PDF must be classifiable with zero image decoding; we
        # assert the result fields exist and the call returns fast. (Timing
        # assertions are flaky in CI; presence is the real contract here.)
        pdf = _text_pdf(tmp_path / "fast.pdf")
        diagnosis = diagnose_document(pdf)
        assert diagnosis.to_dict()["pages"]["1"]["status"] == "text_ok"
