"""CLI for the Darsio ingest engine (debugging tool, not a product surface).

Usage:
    uv run darsio-ingest diagnose data/input/test.pdf
    uv run darsio-ingest process data/input/test.pdf --pages 1-5 --outdir data/output/run
    uv run darsio-ingest bench data/input/test.pdf --questions 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from darsio_ingest.api_types import ProcessOptions
from darsio_ingest.cache import InMemoryPageCache
from darsio_ingest.diagnose import diagnose_document
from darsio_ingest.pipeline import process_document, process_pages
from darsio_ingest.retrieval import LexicalRetriever


def parse_pages(value: str | None) -> list[int] | None:
    """Parse "1,2,5-9" into a sorted list; None means all pages."""
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


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------


def cmd_diagnose(args: argparse.Namespace) -> int:
    diagnosis = diagnose_document(args.pdf, pages=parse_pages(args.pages))

    print(f"Document: {diagnosis.document_id}")
    print(f"Checksum: {diagnosis.checksum_sha256[:16]}...")
    print(f"Pages diagnosed: {len(diagnosis.pages)} / {diagnosis.page_count}")
    print(f"Dominant status: {diagnosis.dominant_status.value}")
    print(f"Recommended method: {diagnosis.recommended_method.value}")
    print()
    print(f"{'page':>4}  {'status':<16} {'method':<20} chars  persian  pua  images  imgs%"
    )
    for number in sorted(diagnosis.pages):
        d = diagnosis.pages[number]
        print(
            f"{number:>4}  {d.status.value:<16} {d.recommended_method.value:<20} "
            f"{d.char_count:>5}  {d.persian_char_count:>7}  {d.pua_char_count:>3}  "
            f"{d.image_count:>6}  {d.image_area_ratio:>5.2f}"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "diagnosis.json"
    out_path.write_text(
        json.dumps(diagnosis.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


def write_outputs(document, outdir: Path, include_blocks: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "document.json").write_text(
        json.dumps(document.to_dict(include_blocks=include_blocks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (outdir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in document.chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    parts = [f"# {document.metadata.document_id}"]
    for page in document.pages:
        parts.append(f"\n--- page {page.page_number} ({page.source_method}) ---\n{page.text}")
    (outdir / "document.txt").write_text("\n\n".join(parts), encoding="utf-8")

    print(f"Wrote {outdir / 'document.json'}")
    print(f"Wrote {outdir / 'chunks.jsonl'}")
    print(f"Wrote {outdir / 'document.txt'}")


def cmd_process(args: argparse.Namespace) -> int:
    pages = parse_pages(args.pages)
    options = ProcessOptions(
        mode=args.mode,
        dpi=args.dpi,
        merge_across_pages=args.merge_across_pages,
        strip_repeated=not args.no_strip_repeated,
    )
    cache = InMemoryPageCache() if args.cache else None

    start = time.perf_counter()
    document = process_document(
        args.pdf,
        pages=pages,
        options=options,
        cache=cache,
    )
    elapsed = time.perf_counter() - start

    write_outputs(document, Path(args.outdir), include_blocks=args.include_blocks)

    summary = {
        "document_id": document.metadata.document_id,
        "pages_processed": len(document.pages),
        "page_count_total": document.metadata.page_count,
        "elapsed_seconds": round(elapsed, 2),
        "chunks": len(document.chunks),
        "method_per_page": {str(k): v for k, v in document.method_per_page.items()},
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "bench.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {outdir / 'bench.json'}")
    print(
        f"\nDone in {elapsed:.1f}s "
        f"({len(document.pages)} pages, {len(document.chunks)} chunks)"
    )
    return 0


# ---------------------------------------------------------------------------
# bench (token/cost estimation, no API calls)
# ---------------------------------------------------------------------------

QUESTIONS = [
    {"question": "ویژگی های فلزات چیست و چه کاربردهایی در زندگی دارند؟", "expected_pages": [3, 4]},
    {"question": "اوزون چیست و چه نقشی دارد؟", "expected_pages": [5, 6]},
    {"question": "کاربردهای آمونیاک کدام اند؟", "expected_pages": [7]},
    {"question": "در مدل بور محل پروتون ها و الکترون ها کجاست؟", "expected_pages": [7, 8]},
    {"question": "در جدول تناوبی ستون ها و سطرها چه نام دارند؟", "expected_pages": [9]},
]


def cmd_bench(args: argparse.Namespace) -> int:
    from darsio_ingest.llm_tokens import LLMCost, ScenarioUsage, estimate_tokens

    start = time.perf_counter()
    document = process_document(
        args.pdf,
        pages=parse_pages(args.pages),
        options=ProcessOptions(dpi=args.dpi),
    )
    ingest_seconds = time.perf_counter() - start

    document_text = "\n\n".join(page.text for page in document.pages)
    retriever = LexicalRetriever(document.chunks)
    cost = LLMCost()

    baseline_tokens = estimate_tokens(document_text)
    per_question = []
    hit_questions = 0
    rag_context_tokens_total = 0

    for item in QUESTIONS[: args.questions]:
        question = item["question"]
        expected = set(item["expected_pages"])
        hits = retriever.search(question, k=args.top_k)
        hit_pages = {
            p
            for hit in hits
            for p in range(hit.chunk.page_start, hit.chunk.page_end + 1)
        }
        page_hit = any(p in expected for p in hit_pages)
        hit_questions += 1 if page_hit else 0

        rag_context = "\n\n".join(hit.chunk.text for hit in hits)
        rag_tokens = estimate_tokens(rag_context)
        rag_context_tokens_total += rag_tokens

        baseline = ScenarioUsage(
            name="baseline",
            context_tokens=baseline_tokens,
            question_tokens=estimate_tokens(question),
            answer_tokens=args.baseline_answer_tokens,
            cost=cost,
        )
        rag = ScenarioUsage(
            name="pipeline_rag",
            context_tokens=rag_tokens,
            question_tokens=estimate_tokens(question),
            answer_tokens=args.rag_answer_tokens,
            cost=cost,
        )
        per_question.append(
            {
                "question": question,
                "expected_pages": sorted(expected),
                "retrieved_pages": sorted(hit_pages),
                "page_hit": page_hit,
                "baseline": baseline.to_dict(),
                "pipeline_rag": rag.to_dict(),
                "input_token_reduction_pct": round(
                    100 * (1 - rag.input_tokens / baseline.input_tokens), 1
                ),
            }
        )
        print(
            f"  Q: {question[:36]}... | baseline_in={baseline.input_tokens} "
            f"rag_in={rag.input_tokens} | hit={page_hit} pages={sorted(hit_pages)}"
        )

    if per_question:
        avg_input_reduction = sum(
            q["input_token_reduction_pct"] for q in per_question
        ) / len(per_question)
    else:
        avg_input_reduction = 0.0

    report = {
        "document_id": document.metadata.document_id,
        "generated_at": time.strftime("%Y-%m-%d"),
        "ingest_seconds": round(ingest_seconds, 2),
        "method_per_page": {str(k): v for k, v in document.method_per_page.items()},
        "config": {
            "top_k": args.top_k,
            "questions": len(per_question),
            "usd_to_irr": cost.usd_to_irr,
            "price_note": "illustrative GPT-4o prices recorded 2026-09-26",
        },
        "baseline_context_tokens": baseline_tokens,
        "questions": per_question,
        "summary": {
            "retrieval_page_hit_rate": round(
                hit_questions / len(per_question), 2
            )
            if per_question
            else 0.0,
            "avg_input_token_reduction_pct": round(avg_input_reduction, 1),
            "rag_context_tokens_avg": round(
                rag_context_tokens_total / len(per_question), 1
            )
            if per_question
            else 0.0,
        },
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "token_benchmark.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nIngest: {ingest_seconds:.1f}s | chunks: {len(document.chunks)}")
    print(f"Retrieval hit-rate: {report['summary']['retrieval_page_hit_rate']}")
    print(f"Avg input token reduction: {report['summary']['avg_input_token_reduction_pct']}%")
    print(f"Wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="darsio-ingest", description="Darsio document-ingestion engine"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_diagnose = sub.add_parser("diagnose", help="Classify pages without OCR")
    p_diagnose.add_argument("pdf")
    p_diagnose.add_argument("--pages", help="e.g. 1,2,5-9 (default: all)")
    p_diagnose.add_argument("--outdir", default="data/output/diagnose")
    p_diagnose.set_defaults(func=cmd_diagnose)

    p_process = sub.add_parser("process", help="Ingest pages (text or OCR)")
    p_process.add_argument("pdf")
    p_process.add_argument("--pages", help="e.g. 1-5 (default: all)")
    p_process.add_argument("--outdir", default="data/output/run")
    p_process.add_argument("--dpi", type=int, default=200)
    p_process.add_argument(
        "--mode", choices=["auto", "text_only", "ocr_only"], default="auto"
    )
    p_process.add_argument(
        "--merge-across-pages", action="store_true", help="Chunks may span pages (textbooks)"
    )
    p_process.add_argument(
        "--no-strip-repeated", action="store_true", help="Keep headers/footers/watermarks"
    )
    p_process.add_argument("--include-blocks", action="store_true")
    p_process.add_argument(
        "--cache", action="store_true", help="Use an in-memory page cache for this run"
    )
    p_process.set_defaults(func=cmd_process)

    p_bench = sub.add_parser("bench", help="Token/cost benchmark (no API calls)")
    p_bench.add_argument("pdf")
    p_bench.add_argument("--pages", help="Restrict ingest to pages")
    p_bench.add_argument("--outdir", default="data/output/bench_tokens")
    p_bench.add_argument("--dpi", type=int, default=200)
    p_bench.add_argument("--top-k", type=int, default=4)
    p_bench.add_argument("--questions", type=int, default=5)
    p_bench.add_argument("--baseline-answer-tokens", type=int, default=350)
    p_bench.add_argument("--rag-answer-tokens", type=int, default=300)
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
