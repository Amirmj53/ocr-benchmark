# darsio-ingest

The document-ingestion engine for
[Darsio](https://github.com/Amirmj53/Darsio): diagnose a PDF, extract the
text layer when it is healthy, OCR only the pages that need it (RapidOCR
PP-OCRv5 Arabic mobile, ONNX Runtime), analyze layout, normalize Persian
conservatively, and emit structure-aware, page-cited chunks ready for RAG.
Also ships the prompt builders for Darsio's three study modes.

Designed for real Persian educational PDFs (scanned pamphlets, 100+ pages)
under a hard economic constraint: never OCR a whole book by default.
~8 s/page and ~1 GB RAM for OCR vs milliseconds for text extraction, so
the engine routes per page and caches everything.

## Package layout (`src/darsio_ingest/`)

| Module | Responsibility |
|---|---|
| `diagnose.py` | page + document classification (text_ok / text_plus_visual / text_broken / scanned / empty), no OCR |
| `extract.py` | text-layer path: pymupdf dict extraction into the same layout pipeline (no OCR) |
| `ocr/` | RapidOCR engine wrapper (process-wide singleton, models load once) + optional preprocessing (off by default, A/B-tested worse) |
| `layout.py` | line grouping, RTL order, columns, regions, header/footer/watermark stripping |
| `normalize/` | conservative Persian normalization (presentation forms, ZWNJ-aware, digits, punctuation) |
| `chunking.py` | structure-aware chunker, page-boundary policy, full RAG metadata |
| `pipeline.py` | dual-path routing, page cache integration, process_pages / process_document / run_ingest_job |
| `cache.py` | PageCache protocol (Darsio wraps it with its DB) + in-memory impl |
| `prompts/` | build_messages + pack_context for normal / exam / research modes |
| `api_types.py` | stable contract types: IngestJob, IngestJobResult, ProcessOptions, CachedPage, PageStatus, ENGINE_VERSION |
| `pdfio.py` | lazy rendering, checksums, document ids |
| `retrieval.py` | lexical IDF retriever (baseline; drop-in interface for embeddings) |
| `cli.py` | `darsio-ingest diagnose / process / bench` (debugging tool) |

How the engine decides, end-to-end, with Darsio integration examples:
see [ENGINE.md](ENGINE.md).

## Quick start

```bash
uv sync --group dev
uv run pytest tests/ -q        # 57 tests, no OCR models needed

# What kind of pages does this PDF have? (no OCR, milliseconds)
uv run darsio-ingest diagnose "data/input/Gilan-College-Pamphelt.pdf"

# Process pages 1-5 only (auto: text path for healthy pages, OCR for scans)
uv run darsio-ingest process "data/input/Gilan-College-Pamphelt.pdf" \
    --pages 1-5 --outdir data/output/run

# Force the OCR path (for scanned pamphlets)
uv run darsio-ingest process data/input/test.pdf --pages 1-3 \
    --mode ocr_only --outdir data/output/ocr

# Token/cost benchmark (no API calls)
uv run darsio-ingest bench data/input/test.pdf --outdir data/output/bench
```

Outputs under `--outdir`: `document.json`, `chunks.jsonl` (embed this),
`document.txt`, `bench.json`, and for diagnose `diagnosis.json`.

## Using it from Darsio

```python
from darsio_ingest import (
    diagnose_pdf, process_pages, build_messages, pack_context,
    run_ingest_job, IngestJob, InMemoryPageCache,
)

# 1) After upload: cheap classification (no OCR, no page decisions)
d = diagnose_pdf("uploads/lecture.pdf")
d.status_counts          # {"text_ok": 12, "scanned": 88, ...}
d.pages_with_status("text_broken", "scanned")  # OCR candidates for quota

# 2) When the user asks about pages 40-52 (product/quota chose the range)
job = IngestJob(document_id="lecture-abc", pdf_path="uploads/lecture.pdf",
                pages=list(range(40, 53)), mode="auto", dpi=200)
result = run_ingest_job(job, cache=InMemoryPageCache())  # swap in your DB cache
result.pages_done, result.method_per_page, result.chunks

# 3) At answer time: build the study-mode prompt
messages = build_messages(
    question="چرا اوزون مهم است؟",
    chunks=result.chunks[:6],
    history=[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    study_mode="exam",       # normal | exam | research
)
# -> OpenAI-style message dicts; call your LLM provider with them
```

The page cache is the only seam Darsio must implement: a `PageCache` with
`get_page_text`, `get_cached_page`, `upsert_page_text` backed by its
SQLAlchemy models. A page is a cache hit only when checksum, dpi, source
method AND engine version all match, so re-running jobs is idempotent and
engine upgrades self-invalidate stale pages. Full examples including the
DB adapter: [ENGINE.md](ENGINE.md).

## Design decisions worth knowing

* **Raw color images beat preprocessing** (A/B tested): grayscale +
  contrast + sharpen destroyed a real content line and produced
  letter-spam. Preprocessing stays available but off.
* **Uncertain content is classified, not deleted**: low-score formula-like
  content is kept down to 0.30 confidence as SPECIAL; garbage-looking text
  (letter-spam, decorative marks, footer banners) is what gets removed.
* **Repeated headers/footers/watermarks** are removed by cross-page text
  repetition (>= 60% of pages), never by fixed positions alone.
* **RTL correctness**: only the *order of boxes* is mapped to reading
  order (right to left); Persian strings themselves are never reversed.
* **Arabic presentation forms** (FB50-FDFF, FE70-FEFF) are folded to
  standard Persian letters during normalization; diagnose counts them as
  healthy Persian, not broken text.
* **Page-boundary policy**: chunks don't span pages by default
  (slide-like pamphlets); pass `merge_across_pages=True` for flowing
  textbooks, and `reset_sections_per_page=False` to keep section context.
* **Normalization is conservative**: ZWNJ is meaningful and kept; Arabic
  letters fold to Persian; digits fold to ASCII; diacritics are stripped;
  nothing alphanumeric is ever deleted.
* **RapidOCR over PaddleOCR** was a measured benchmark win on the
  reference machine; PaddleOCR was removed from dependencies. Do not swap
  engines without a documented A/B win.

## Benchmarks (reference machine, 200 DPI)

| Path | Speed | Notes |
|---|---|---|
| diagnose | ~ms / page | object-level only, no rendering |
| text path | ~ms / page | healthy text layers |
| OCR path | ~7.9 s / page | RapidOCR PP-OCRv5 Arabic mobile |
| RAM peak (OCR) | ~1 GB | one page image at a time, models loaded once |

RAG retrieval on the 13-page chemistry pamphlet (lexical retriever,
k=4): 5/5 page hit-rate, ~80% input token reduction vs sending the full
OCR text.

## Reference

Ideas on Persian OCR configuration, normalization and RTL handling were
inspired by (not copied from)
[Hamed-Gharghi/Persian-OCR-App](https://github.com/Hamed-Gharghi/Persian-OCR-App).
