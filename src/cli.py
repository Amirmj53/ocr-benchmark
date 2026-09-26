"""CLI for the Darsio OCR pipeline.

Usage:
    uv run python src/cli.py data/input/test.pdf                # pages 1-13
    uv run python src/cli.py data/input/test.pdf --pages 2,5,7
    uv run python src/cli.py data/input/test.pdf --outdir data/output/run1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import PipelineConfig, run_pipeline  # noqa: E402


def parse_pages(value: str | None) -> list[int] | None:
    if not value:
        return None
    pages: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


def write_outputs(run, outdir: Path, include_blocks: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    document = run.document

    # Full structured JSON.
    json_path = outdir / "document.json"
    json_path.write_text(
        json.dumps(document.to_dict(include_blocks=include_blocks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Chunks as JSONL — one chunk per line, ready for embeddings.
    jsonl_path = outdir / "chunks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for chunk in document.chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    # Human-readable text.
    text_path = outdir / "document.txt"
    parts = [f"# {document.metadata.document_id}"]
    for page in document.pages:
        parts.append(f"\n--- page {page.page_number} ---\n{page.text}")
    text_path.write_text("\n\n".join(parts), encoding="utf-8")

    # Bench summary.
    summary = {
        "document_id": document.metadata.document_id,
        "pages_processed": len(document.pages),
        "page_count_total": document.metadata.page_count,
        "elapsed_seconds": round(run.elapsed_seconds, 2),
        "ocr_seconds": round(run.ocr_seconds, 2),
        "avg_page_seconds": round(run.elapsed_seconds / max(len(document.pages), 1), 2),
        "repeated_texts_removed": sorted(run.repeated_texts),
        "chunks": len(document.chunks),
    }
    (outdir / "bench.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote {json_path}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {text_path}")
    print(f"Wrote {outdir / 'bench.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Darsio OCR pipeline")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--pages", help="Pages to process, e.g. 1,2,5-9 (default: all)")
    parser.add_argument("--outdir", default="data/output/run", help="Output directory")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-strip-repeated", action="store_true", help="Keep headers/footers/watermarks")
    parser.add_argument("--keep-noise", action="store_true", help="Keep noise blocks in output")
    parser.add_argument("--include-blocks", action="store_true", help="Include raw OCR blocks in JSON")
    parser.add_argument("--save-page-images", action="store_true", help="Dump rendered page images")
    args = parser.parse_args(argv)

    config = PipelineConfig(
        dpi=args.dpi,
        strip_repeated=not args.no_strip_repeated,
        keep_noise_blocks=args.keep_noise,
        save_debug_images=args.save_page_images,
    )

    pages = parse_pages(args.pages)
    start = time.perf_counter()
    run = run_pipeline(args.pdf, config=config, page_numbers=pages)
    write_outputs(run, Path(args.outdir), include_blocks=args.include_blocks)

    print(
        f"\nDone in {time.perf_counter() - start:.1f}s "
        f"({len(run.document.pages)} pages, {len(run.document.chunks)} chunks)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
