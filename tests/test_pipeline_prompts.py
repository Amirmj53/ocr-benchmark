"""Smoke tests for prompts and the dual-path pipeline on fixture PDFs.

The OCR path is deliberately NOT exercised here (models are heavy); it is
covered by the manual CLI workflow on real scans. Everything else --
routing, caching, chunking, prompt assembly -- runs in milliseconds.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from darsio_ingest.api_types import CachedPage, ProcessOptions
from darsio_ingest.cache import InMemoryPageCache
from darsio_ingest.chunking import chunk_regions
from darsio_ingest.models import Chunk, ChunkType, TextRegion
from darsio_ingest.pipeline import process_document, process_pages
from darsio_ingest.prompts import build_messages, pack_context


def make_text_pdf(path: Path, pages: int = 3) -> Path:
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        # insert_htmlbox embeds a real Unicode font so the Persian text
        # layer actually extracts (helv textbox emits '?' spam).
        page.insert_htmlbox(
            pymupdf.Rect(72, 72, 540, 700),
            f"<p>فصل {i + 1}. این یک متن آزمایشی فارسی است. " * 6 + "</p>",
        )
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Chunk schema
# ---------------------------------------------------------------------------


def _chunk(index: int, page: int, text: str, method: str = "ocr") -> Chunk:
    return Chunk(
        chunk_id=f"doc-c{index:04d}",
        document_id="doc",
        chunk_index=index,
        chunk_type=ChunkType.PARAGRAPH,
        text=text,
        page_start=page,
        page_end=page,
        char_count=len(text),
        source_method=method,
    )


class TestPresentationForms:
    def test_presentation_form_text_is_text_ok(self, tmp_path):
        # htmlbox writes Persian as Arabic presentation forms (FB50-FDFF);
        # the diagnose layer must accept them as healthy Persian text.
        pdf = make_text_pdf(tmp_path / "pf.pdf", pages=1)
        from darsio_ingest.diagnose import diagnose_document

        diagnosis = diagnose_document(pdf)
        assert diagnosis.pages[1].status.value == "text_ok"

    def test_extracted_text_is_standard_persian(self, tmp_path):
        # After normalization, presentation forms fold to standard letters.
        pdf = make_text_pdf(tmp_path / "pf.pdf", pages=1)
        results = process_pages(pdf, pages=[1])
        text = results[0].text
        assert len(text) > 100
        assert not any(0xFB50 <= ord(c) <= 0xFDFF for c in text)


class TestPresentationFormsUnit:
    def test_normalize_folds_presentation_forms(self):
        from darsio_ingest.normalize import normalize_text

        #ﺍ (U+FE8D, ALEF final form) -> ا ; ﻡ (U+FE4D... variant) safe
        folded = normalize_text("\uFE8D\uFEE3")
        assert folded == "ام"

    def test_is_persian_char_accepts_presentation_forms(self):
        from darsio_ingest.normalize import is_persian_char

        assert is_persian_char("\uFE8D")
        assert is_persian_char("ا")
        assert not is_persian_char("a")


class TestChunkSchema:
    def test_new_provenance_fields_present(self):
        chunk = _chunk(0, 1, "متن")
        data = chunk.to_dict()
        assert data["source_method"] == "ocr"
        assert data["engine_version"]
        assert data["normalized"] is True

    def test_page_bounds_default_consistent(self):
        chunk = _chunk(0, 7, "متن")
        assert chunk.page_start == chunk.page_end == 7


class TestPackContext:
    def test_headers_and_deterministic_order(self):
        chunks = [
            _chunk(1, 5, "دوم"),
            _chunk(0, 3, "اول"),
        ]
        context = pack_context(chunks, max_chars=10_000)
        assert context.index("اول") < context.index("دوم")
        assert "[صفحه 3" in context
        assert "[صفحه 5" in context

    def test_respects_char_budget(self):
        chunks = [_chunk(i, i + 1, "x" * 500) for i in range(10)]
        context = pack_context(chunks, max_chars=800)
        assert len(context) < 800 + 600  # at least one chunk always fits
        assert context.count("[صفحه") == 1

    def test_multi_page_header(self):
        chunk = _chunk(0, 3, "متن")
        chunk.page_end = 4
        context = pack_context([chunk], max_chars=1000)
        assert "صفحه‌های 3-4" in context


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_shape_and_mode_dispatch(self):
        chunks = [_chunk(0, 1, "منبع آزمایشی")]
        for mode in ("normal", "exam", "research"):
            messages = build_messages("سؤال؟", chunks, None, study_mode=mode)
            assert messages[0]["role"] == "system"
            assert messages[-1]["role"] == "user"
            assert "<document_context>" in messages[-1]["content"]
            assert "منبع آزمایشی" in messages[-1]["content"]

    def test_invalid_mode_raises(self):
        try:
            build_messages("q", [], None, study_mode="chat")
        except ValueError as error:
            assert "study_mode" in str(error)
        else:
            raise AssertionError("expected ValueError")

    def test_history_sanitized(self):
        messages = build_messages(
            "سؤال",
            [],
            history=[
                {"role": "user", "content": "قبلی"},
                {"role": "tool", "content": "junk"},
                {"role": "assistant", "content": "پاسخ"},
            ],
            study_mode="normal",
        )
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]

    def test_mode_prompts_differ(self):
        from darsio_ingest.prompts import (
            EXAM_SYSTEM_PROMPT,
            NORMAL_SYSTEM_PROMPT,
            RESEARCH_SYSTEM_PROMPT,
        )

        prompts = {NORMAL_SYSTEM_PROMPT, EXAM_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT}
        assert len(prompts) == 3

    def test_citation_rule_present_in_all_modes(self):
        from darsio_ingest.prompts import (
            EXAM_SYSTEM_PROMPT,
            NORMAL_SYSTEM_PROMPT,
            RESEARCH_SYSTEM_PROMPT,
        )

        for prompt in (NORMAL_SYSTEM_PROMPT, EXAM_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT):
            assert "صفحه" in prompt
            assert "منبع" in prompt


# ---------------------------------------------------------------------------
# Pipeline: text path + cache
# ---------------------------------------------------------------------------


class TestProcessPagesTextPath:
    def test_text_pdf_never_ocrs(self, tmp_path):
        pdf = make_text_pdf(tmp_path / "t.pdf", pages=2)
        results = process_pages(pdf, pages=[1, 2])
        assert [r.page_number for r in results] == [1, 2]
        for result in results:
            assert result.source_method == "text"
            assert not result.fallback_used
            assert len(result.text) > 100

    def test_page_selection(self, tmp_path):
        pdf = make_text_pdf(tmp_path / "t.pdf", pages=4)
        results = process_pages(pdf, pages=[2])
        assert [r.page_number for r in results] == [2]

    def test_range_selection(self, tmp_path):
        pdf = make_text_pdf(tmp_path / "t.pdf", pages=4)
        results = process_pages(pdf, pages=range(1, 3))
        assert [r.page_number for r in results] == [1, 2]

    def test_chunks_have_provenance(self, tmp_path):
        pdf = make_text_pdf(tmp_path / "t.pdf", pages=1)
        results = process_pages(pdf, pages=[1])
        assert results[0].chunks
        for chunk in results[0].chunks:
            assert chunk.source_method == "text"
            assert chunk.engine_version
            assert chunk.normalized is True

    def test_cache_hit_skips_work(self, tmp_path):
        pdf = make_text_pdf(tmp_path / "t.pdf", pages=1)
        cache = InMemoryPageCache()

        first = process_pages(pdf, pages=[1], cache=cache)
        assert not first[0].cache_hit
        assert len(cache) == 1

        second = process_pages(pdf, pages=[1], cache=cache)
        assert second[0].cache_hit
        assert second[0].text == first[0].text

    def test_cache_miss_on_checksum_change(self, tmp_path):
        pdf_a = make_text_pdf(tmp_path / "a.pdf", pages=1)
        cache = InMemoryPageCache()
        process_pages(pdf_a, pages=[1], cache=cache, document_id="doc")

        pdf_b = make_text_pdf(tmp_path / "b.pdf", pages=1)  # different bytes
        results = process_pages(pdf_b, pages=[1], cache=cache, document_id="doc")
        assert not results[0].cache_hit


class TestCachedPageIdentity:
    def test_matches_requires_full_identity(self):
        page = CachedPage(
            document_id="doc",
            page_no=1,
            text="t",
            checksum_sha256="aaa",
            dpi=200,
            source_method="text",
        )
        assert page.matches("aaa", 200, "text")
        assert not page.matches("bbb", 200, "text")
        assert not page.matches("aaa", 300, "text")
        assert not page.matches("aaa", 200, "ocr")

    def test_roundtrip_dict(self):
        page = CachedPage(
            document_id="doc", page_no=2, text="متن", checksum_sha256="x"
        )
        restored = CachedPage.from_dict(page.to_dict())
        assert restored == page


class TestProcessDocument:
    def test_document_result_shape(self, tmp_path):
        pdf = make_text_pdf(tmp_path / "t.pdf", pages=3)
        document = process_document(pdf)
        assert document.metadata.page_count == 3
        assert document.metadata.checksum_sha256
        assert len(document.pages) == 3
        assert document.chunks
        assert document.method_per_page == {1: "text", 2: "text", 3: "text"}
        assert all(c.source_method == "text" for c in document.chunks)

    def test_merge_across_pages_option(self, tmp_path):
        pdf = make_text_pdf(tmp_path / "t.pdf", pages=3)
        merged = process_document(pdf, options=ProcessOptions(merge_across_pages=True))
        default = process_document(pdf)
        # Both produce chunks; merged output may join page-boundary chunks.
        assert merged.chunks
        assert default.chunks
        assert any(c.page_end > c.page_start for c in merged.chunks) or any(
            c.page_end > c.page_start for c in default.chunks
        )
