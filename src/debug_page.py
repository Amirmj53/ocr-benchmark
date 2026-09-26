"""Diagnostics: dump raw OCR blocks + line classification for one page."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layout import (
    MIN_BLOCK_SCORE,
    classify_line,
    group_into_lines,
    is_noise_block,
    looks_like_formula,
    order_lines_reading_order,
    statistics_median,
)
from models import LineType
from normalize import normalize_text
from ocr import OCREngine
from pdf import iter_pages
from preprocessing import preprocess_image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("page", type=int)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    engine = OCREngine()
    page_input = next(iter_pages(args.pdf, dpi=args.dpi, page_numbers=[args.page]))
    image = preprocess_image(page_input.image)
    blocks, _ = engine.run_page(image, page_input.page_number)

    print(f"RAW BLOCKS ({len(blocks)}) — page {args.page}")
    print("=" * 100)
    for index, block in enumerate(sorted(blocks, key=lambda b: (b.top, -b.center_x))):
        noise = is_noise_block(block)
        low = block.score < MIN_BLOCK_SCORE
        flags = ("NOISE" if noise else "     ") + (" LOW" if low else "") + (" F?" if looks_like_formula(block.text) else "")
        print(
            f"{index:02d} | s={block.score:.2f} | {flags:<14} | "
            f"y={block.top:6.0f} x={block.center_x:6.0f} | {block.text}"
        )

    kept = [b for b in blocks if b.score >= MIN_BLOCK_SCORE and not is_noise_block(b)]
    lines = group_into_lines(kept)
    ordered, columns = order_lines_reading_order(lines, float(page_input.image.size[0]))
    median_height = statistics_median([line.height for line in ordered])

    print(f"\nLINES ({len(ordered)}), columns={columns}, median_h={median_height:.0f}")
    print("=" * 100)
    for index, line in enumerate(ordered):
        line_type = classify_line(line, median_height, body_width_hint=max(l.width for l in ordered))
        marker = {"body": "    ", "heading": "HEAD", "special": "SPEC", "noise": "NZ"}[line_type.value]
        print(
            f"{index + 1:02d} {marker} | y={line.top:6.0f} h={line.height:5.0f} "
            f"w={line.width:5.0f} | {line.text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
