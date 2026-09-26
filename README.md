# Darsio OCR Pipeline (درسیو)

PDF → render → RapidOCR (PP-OCRv5 Arabic) → layout analysis → Persian
normalization → structure-aware chunking → retrieval-ready JSON/JSONL.

Designed for real Persian educational PDFs (scanned pamphlets, 100+ pages)
and built as the ingestion foundation for search, embeddings, RAG, question
generation and page-cited answers in Darsio.

## Pipeline

```
PDF (pymupdf, lazy page rendering, 200 DPI)
  → OCR (RapidOCR, PP-OCRv5 Arabic mobile, ONNX Runtime)
  → blocks        raw boxes + text + confidence + coordinates
  → lines         vertical-overlap grouping, RTL block ordering
  → reading order single/two-column detection (right column first)
  → regions       title / heading / body / special / noise + paragraphs
  → page cleanup  headers/footers/watermarks, decorative marks, letter-spam
  → normalize     careful Persian normalization (never destroys content)
  → chunks        heading sections, sentence-boundary splits, full metadata
```

## Modules (`src/`)

| Module | Responsibility |
|---|---|
| `pdf.py` | rendering (lazy per-page), metadata, stable document ids |
| `ocr.py` | RapidOCR engine wrapper, in-memory images, block extraction |
| `preprocessing.py` | optional grayscale/contrast/sharpen (off by default — A/B tested slower *and* worse) |
| `layout.py` | line grouping, columns/reading order, classification, repeated-content removal, page assembly |
| `normalize.py` | Persian letter folding, diacritics, digits, punctuation, ZWNJ-aware |
| `chunking.py` | structure-aware chunker with page-boundary policy |
| `models.py` | OCRBlock / OCRLine / TextRegion / PageResult / Chunk / DocumentResult |
| `pipeline.py` | orchestration + per-page timings |
| `cli.py` | command-line entrypoint, writes JSON / JSONL / TXT / bench |
| `debug_page.py` | raw OCR + classification diagnostics for one page |
| `test_other_docs.py` | generalization check on other PDFs |

## Usage

```bash
# Full document (13 pages of the benchmark PDF)
uv run python src/cli.py data/input/test.pdf --outdir data/output/run

# Selected pages only (fast iteration)
uv run python src/cli.py data/input/test.pdf --pages 7 --outdir data/output/p7

# Different DPI, keep headers/footers, keep noise blocks in JSON
uv run python src/cli.py data/input/test.pdf --dpi 300 \
    --no-strip-repeated --keep-noise --include-blocks

# Generalization check on any other PDF
uv run python src/test_other_docs.py "data/input/Gilan-College-Pamphelt.pdf" --pages 4,5

# Single-page OCR/layout diagnostics
uv run python src/debug_page.py data/input/test.pdf 7
```

Outputs in `--outdir`:

* `document.json` — full structure: metadata, pages, regions (bboxes, roles, scores), chunks
* `chunks.jsonl` — one retrieval-ready chunk per line (embed this)
* `document.txt` — human-readable per-page text
* `bench.json` — timing summary + removed repeated texts

## Chunk metadata (per chunk)

`chunk_id`, `document_id`, `chunk_index`, `chunk_type`
(paragraph/heading/special), `text`, `page_start`, `page_end`, `section`,
`section_path`, `bbox` (union geometry), `score` (mean OCR confidence),
`language`, `char_count` — everything needed for page-cited RAG.

## Benchmark (this machine, 13-page scanned PDF, 200 DPI)

| Stage | Time |
|---|---|
| Total pipeline | ~103 s (~7.9 s/page) |
| OCR (RapidOCR PP-OCRv5 Arabic mobile) | ~98 s |
| Layout + chunking + I/O | ~5 s |

RapidOCR was chosen over PaddleOCR after benchmarking: significantly faster
on this machine at comparable quality. Do not swap engines without a
documented A/B win.

Example: `data/output/final/` contains the final run for `data/input/test.pdf`.

## Design decisions worth knowing

* **Raw color images beat preprocessing** on the benchmark (A/B tested):
  grayscale+contrast+sharpen destroyed a real content line and produced
  letter-spam. Preprocessing stays available but off.
* **Uncertain content is classified, not deleted**: low-score formula-like
  content is kept down to 0.30 confidence as SPECIAL; garbage-looking text
  (letter-spam, decorative marks, footer banners) is what gets removed.
* **Repeated headers/footers/watermarks** are removed by cross-page text
  repetition (≥60% of pages), never by fixed positions alone.
* **RTL correctness**: only the *order of blocks* is mapped to reading
  order (right→left); Persian strings themselves are never reversed.
* **Page-boundary policy**: chunks don't span pages by default (slide-like
  pamphlets); pass `merge_across_pages=True` in `chunk_regions` for flowing
  textbooks, and `reset_sections_per_page=False` to keep section context.
* **Normalization is conservative**: ZWNJ is meaningful and kept; Arabic
  letters fold to Persian; digits fold to ASCII; diacritics are stripped;
  nothing alphanumeric is ever deleted.

## Reference

Ideas on Persian OCR configuration, normalization and RTL handling were
inspired by (not copied from)
[Hamed-Gharghi/Persian-OCR-App](https://github.com/Hamed-Gharghi/Persian-OCR-App).
