"""Generalization test: run the pipeline on pages from OTHER PDFs.

Prevents overfitting the layout/chunking rules to one document.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import PipelineConfig, run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--pages", required=True, help="e.g. 1,2,3")
    args = parser.parse_args()

    pages = [int(p) for p in args.pages.split(",")]
    run = run_pipeline(args.pdf, PipelineConfig(), page_numbers=pages)

    for page in run.document.pages:
        print(f"\n=== PAGE {page.page_number} ===  columns={page.column_count}")
        for region in page.regions:
            print(f"  {region.role.value:<8} | {region.text[:100]}")

    print(f"\n--- CHUNKS ({len(run.document.chunks)}) ---")
    for chunk in run.document.chunks:
        print(
            f"  {chunk.chunk_index:02d} p{chunk.page_start}-{chunk.page_end} "
            f"{chunk.chunk_type.value:<9} {chunk.char_count:4d} sec={chunk.section}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
