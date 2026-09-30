# Darsio Ingest Engine

The document-ingestion engine behind Darsio's chat. This package owns
everything between "user uploaded a PDF" and "retrievable, page-cited
chunks": diagnosis, dual-path text acquisition (extract or OCR), layout
analysis, Persian normalization, structure-aware chunking, and the prompt
builders for the three study modes.

It is deliberately NOT a web service. It is a library plus a debugging CLI.
Darsio's FastAPI app imports it, wraps the page cache in its DB, and keeps
all HTTP, quota and LLM concerns on its side.

---

## 1. Customer path: from upload to chunks

```
User uploads PDF (Darsio POST /documents)
  |
  v
Darsio stores file under uploads/, computes nothing yet
  |
  v
 diagnose_pdf(path)                      [cheap: no render, no OCR, ~ms/page]
  |   returns per-page: text_ok | text_plus_visual | text_broken |
  |                     scanned | empty
  |   plus recommended method per page: text_extract_sorted | ocr | hybrid
  |
  v
Darsio product/quota layer decides WHICH pages to process
  (e.g. user asks about page range 40-52; quota allows OCR of 13 pages)
  |
  v
run_ingest_job(IngestJob(...)) or process_pages(path, pages=...)
  |
  v
per page, the engine routes: healthy text -> extract; else -> OCR
  |
  v
chunks (with page numbers, section path, source method) go to Darsio's
index; page text goes into the page cache for later reuse
  |
  v
at question time: Darsio retrieves top-k chunks,
prompts.build_messages(question, chunks, history, study_mode),
calls its LLM, meters usage
```

Key economic property: OCR is the expensive path (~8 s/page, ~1 GB RAM on
the reference machine). The engine makes OCR the exception, not the rule:
any page with a healthy text layer is processed in milliseconds.

## 2. Diagnose decision tree

`diagnose_pdf(path, pages=None)` reads each page's objects with pymupdf and
computes char counts, Persian-script share (including Arabic presentation
forms FB50-FDFF and FE70-FEFF), broken-char share (PUA, U+FFFD, control
spam), text blocks, images with placed-area coverage, and vector-drawing
count. Then, per page:

```
no text AND no images AND no drawings      -> empty
char_count < 50:
    broken-char ratio >= 10%               -> text_broken
    no images and no drawings              -> empty (zero chars) / scanned
    otherwise                              -> scanned
broken-char ratio >= 10%                   -> text_broken
fewer than 20 letter chars (Persian+Latin) -> text_broken  (typographic ghost)
images cover >= 30% of page area, or
    image present, or >= 20 drawings       -> text_plus_visual
otherwise                                  -> text_ok
```

Method recommendation: `text_ok -> text_extract_sorted`,
`text_plus_visual -> hybrid`, `text_broken -> ocr`, `scanned -> ocr`,
`empty -> skip`.

Document-level diagnosis aggregates: dominant status, status counts, and a
conservative document method (any OCR-needing page + any text page ->
hybrid). Diagnose NEVER decides ranges -- the product layer does.

## 3. Text path vs OCR path

| | text path | OCR path |
|---|---|---|
| Trigger | `text_ok`, or `text_plus_visual` without `ocr_hybrid_pages`, or explicit `mode="text_only"` | `text_broken`, `scanned`, `ocr_only`, or auto-fallback when extraction is silently empty (< 40 chars) |
| Reader | pymupdf `get_text("dict", sort=True)` | lazy render at `options.dpi` (default 200) |
| Lines | spans -> OCRBlock(score=1.0) -> same layout code | RapidOCR PP-OCRv5 Arabic mobile -> OCRBlock(score) |
| Speed | ~ms per page | ~8 s per page |
| Cost guard | never runs OCR models | engine is a process-wide singleton; models load once |

Both paths converge on the same layout -> normalize -> chunk machinery, so
downstream code cannot tell them apart except via `source_method`
(`"text"` | `"ocr"`) recorded on pages, regions and chunks.

Auto-fallback guard: a page diagnosed `text_ok` whose extraction still
yields < `min_text_chars` (default 40) is re-routed to OCR. This is what
protects against fonts whose text layer extracts as `?`-spam.

## 4. Layout, normalize, chunk

**Layout** (identical for both paths):
blocks -> filter (score >= 0.55; formulas kept down to 0.30; noise rules)
-> lines grouped by vertical overlap -> RTL order within a line
(rightmost block first; Persian strings are NEVER reversed) -> 1 vs 2
column detection -> reading order (right column top-down, then left)
-> line classification (heading/body/special/noise via height, width,
light-verb heuristics, footer markers) -> regions -> paragraphs.
Cross-page: lines repeated on >= 60% of pages (headers/footers/watermarks)
are stripped once the batch has >= 4 pages.

**Normalize** (conservative; form changes, never content changes):
NFC -> fold Arabic presentation forms per-char via NFKC -> drop zero-width
marks except ZWNJ -> Arabic letters fold to Persian (yeh/kaf/...) -> strip
diacritics and tatweel -> Persian/Arabic-Indic digits to ASCII -> Persian
punctuation to ASCII -> OCR space-before-punctuation fix -> collapse
spaces. Nothing alphanumeric is ever deleted; ZWNJ is preserved; strings
are never reordered.

**Chunking** (structure-aware):
each heading opens a section (section_path capped at 3 levels, running
titles do not grow the path); body accumulates; oversize pieces split at
Persian sentence enders (`.`, `؟`, `!`), then commas, then hard cut; tiny
remnants merge forward; specials (formulas, chart labels) merge with
neighbors or coalesce; a heading with no body is still emitted as a
heading chunk. Default policy: chunks do NOT span pages (pamphlets); pass
`merge_across_pages=True` for flowing textbooks. Every chunk carries the
full RAG schema:

```
chunk_id, document_id, chunk_index, chunk_type, text,
page_start, page_end, section, section_path, bbox, page_bboxes,
score, language, char_count, source_method, engine_version, normalized
```

## 5. Calling the engine from Darsio (FastAPI)

Install once (this repo stays a separate package; pin by path or git):

```toml
# Darsio pyproject.toml
[project]
dependencies = [
    "darsio-ingest @ file:///srv/darsio/vendor/darsio_ingest-0.2.0-py3-none-any.whl",
]
```

### 5.1 Upload-time diagnosis (router stays thin)

```python
# app/services/ai/ingest/diagnose.py
from darsio_ingest import diagnose_pdf

def diagnose_upload(pdf_path: str) -> dict:
    d = diagnose_pdf(pdf_path)
    return {
        "document_id": d.document_id,
        "page_count": d.page_count,
        "status_counts": d.status_counts,
        "dominant_status": d.dominant_status.value,
        "recommended_method": d.recommended_method.value,
        # pages the user can be offered for OCR, if quota allows:
        "ocr_candidate_pages": d.pages_with_status("text_broken", "scanned"),
    }
```

### 5.2 Processing a page range at question time

```python
# app/services/ai/ingest/ocr_worker.py
from darsio_ingest import (
    IngestJob, PageStatus, process_pages, run_ingest_job,
)

def ingest_page_range(document_id: str, pdf_path: str, pages: list[int]):
    job = IngestJob(
        document_id=document_id,
        pdf_path=pdf_path,
        pages=pages,           # product/quota decided this, not the engine
        mode="auto",           # or "text_only" / "ocr_only"
        dpi=200,
    )
    result = run_ingest_job(job, cache=DBPageCache(session))
    # result.pages_done, result.method_per_page, result.chunks
    for chunk in result.chunks:
        session.add(ChunkRow(**chunk.to_dict()))
    session.commit()
```

### 5.3 DB-backed page cache (the idempotency seam)

```python
# app/models/page_cache.py  (SQLAlchemy mirror of CachedPage)
class PageCacheRow(Base):
    __tablename__ = "page_cache"
    document_id: Mapped[str]
    page_no: Mapped[int]
    text: Mapped[str]
    checksum_sha256: Mapped[str]
    dpi: Mapped[int]
    source_method: Mapped[str]
    engine_version: Mapped[str]
    created_at: Mapped[datetime]

# app/services/ai/ingest/cache.py
from darsio_ingest import CachedPage, PageCache

class DBPageCache(PageCache):
    def __init__(self, session): self.session = session

    def get_page_text(self, document_id, page_no):
        row = self.session.get(PageCacheRow, (document_id, page_no))
        return row.text if row else None

    def get_cached_page(self, document_id, page_no):
        row = self.session.get(PageCacheRow, (document_id, page_no))
        if row is None:
            return None
        return CachedPage(
            document_id=row.document_id, page_no=row.page_no, text=row.text,
            checksum_sha256=row.checksum_sha256, dpi=row.dpi,
            source_method=row.source_method, engine_version=row.engine_version,
            created_at=row.created_at.isoformat(),
        )

    def upsert_page_text(self, page: CachedPage):
        row = self.session.get(PageCacheRow, (page.document_id, page.page_no))
        if row is None:
            row = PageCacheRow(**page.to_dict()); self.session.add(row)
        else:
            for k, v in page.to_dict().items(): setattr(row, k, v)
```

Idempotency: the engine calls `get_cached_page` and treats a page as done
only when checksum + dpi + source_method + engine_version ALL match
(`CachedPage.matches`). After an engine upgrade, old rows are misses and
pages reprocess once.

### 5.4 Prompts at answer time

```python
# app/services/ai/orchestrator.py
from darsio_ingest import build_messages, pack_context, process_pages

def answer(question, document_id, study_mode, session):
    chunks = retrieve_top_k(session, document_id, question, k=6)
    messages = build_messages(
        question=question,
        chunks=[to_engine_chunk(c) for c in chunks],
        history=load_history(session),
        study_mode=study_mode,        # "normal" | "exam" | "research"
        max_context_chars=6000,
    )
    return llm.chat(messages)         # Darsio's provider layer
```

`build_messages` returns OpenAI-style dicts: system (mode-specific Persian
grounding contract), sanitized history, then user = question +
`<document_context>` with deterministic `[صفحه N | متن/OCR | بخش: ...]`
blocks packed to the char budget without mid-chunk cuts.

## 6. Queue / worker recommendations

* **1 worker process** for OCR. The RapidOCR engine is a process-wide
  singleton (~1 GB RAM, ~8 s/page); two workers double peak RAM and do not
  double throughput on one machine. Run ingest OUT of the API process
  (separate `python -m app.workers.ingest` loop or a thread on a tiny
  deployment) so uploads never block on model load.
* **Page limits**: cap pages per job (suggest 16) and OCR pages per user
  per day at the quota layer. `text_only` jobs are cheap enough to be
  unquotaed; treat `text_plus_visual` as text cost by default.
* **Idempotency**: re-driving the same job is free for cached pages; the
  worker can crash mid-document and resume.
* **RAM**: one page image at a time (lazy rendering). Do not call
  `load_document` on 100-page scans in the worker.
* **Engine version**: it is part of the cache identity. Bump
  `ENGINE_VERSION` when layout/chunking/OCR changes so caches self-invalidate.

## 7. Intentionally NOT in v1

* No full-book automatic OCR. Scanned books are OCR'd page-range by
  page-range on demand, under quota.
* No embeddings server, no vector store. Retrieval here is a lexical IDF
  baseline; Darsio plugs its own backend behind `search()`.
* No HTTP API in this package. Library + CLI only.
* No LLM calls. `prompts/` builds messages; Darsio owns the provider,
  usage metering and history persistence.
* No PDF generation, export, or UI.
* No multi-file packaging of OCR alternatives (PaddleOCR removed; the
  engine choice is fixed by documented benchmark).

---

## خلاصه فارسی

این بسته، موتور ورودیِ سند برای چت درسیو است: تشخیص نوع صفحه، دو مسیر
دریافت متن (استخراج یا OCR)، تحلیل چیدمان، نرمال‌سازی فارسی، چانک‌بندی
ساختارمند و پرامپت‌ساز سه حالت مطالعه.

**مسیر کاربر**: آپلود PDF → `diagnose_pdf` (ارزان، بدون OCR) → لایهٔ سهمیهٔ
درسیو تصمیم می‌گیرد کدام صفحات پردازش شوند → `run_ingest_job` یا
`process_pages` → چانک‌ها با شماره صفحه به ایندکس می‌روند و متن صفحه‌ها در
کش ذخیره می‌شود → در زمان سؤال، top-k چانک بازیابی و با
`build_messages(question, chunks, history, study_mode)` پرامپت ساخته می‌شود.

**تصمیم تشخیص**: صفحه با متن سالم یونیکد → `text_ok` (استخراج متن، بدون
OCR)؛ متن + گرافیک سنگین → `text_plus_visual` (hybrid)؛ متن خراب (PUA یا
کاراکتر کنترلی) → `text_broken` (OCR)؛ بدون متن و پُر از تصویر →
`scanned` (OCR)؛ هیچ → `empty`. تشخیص هرگز محدودهٔ صفحات را تعیین نمی‌کند.

**دو مسیر**: استخراجِ متن چند میلی‌ثانیه است؛ OCR حدود ۸ ثانیه بر صفحه و
~۱ گیگابایت حافظه. موتور OCR یک بار در کل فرایند بارگذاری می‌شود
(singleton) و صفحات تک‌به‌تک رندر می‌شوند. اگر صفحهٔ «متن‌دار» عملاً خالی
استخراج شد، خودکار به OCR برمی‌گردد.

**کش صفحه‌ها**: خطی با document_id، شماره صفحه، checksum، dpi، روش منبع و
نسخهٔ موتور. تکرار همان درخواست = بی‌کار (idempotent). نسخهٔ موتور بخشی از
کلید کش است؛ ارتقای موتور، کش کهنه را باطل می‌کند.

**توصیه‌های صف**: یک ورکر برای OCR، سقف ۱۶ صفحه در هر job، سهمیهٔ روزانه
برای صفحات OCR در لایهٔ محصول، اجرای ingest خارج از فرایند FastAPI.

**عمداً در نسخهٔ اول نیست**: OCR خودکار کل کتاب، سرور embedding، API وب،
فراخوانی LLM. این بسته فقط ingest + پرامپت‌ساز + تایپ‌هاست.
