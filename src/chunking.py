"""Structure-aware Persian chunking.

Chunks follow document structure instead of slicing characters:

* each heading starts a new section;
* paragraphs accumulate inside the current section;
* long paragraphs are split at sentence boundaries (Persian ``.``, ``؟``,
  ``!``), never mid-word;
* tiny remnants are merged forward so we never emit fragment chunks;
* every chunk carries document_id, page_start/page_end, section/heading,
  section_path, chunk_index, geometry and OCR confidence.
"""

from __future__ import annotations

import re

from models import Chunk, ChunkType, DocumentResult, RegionRole, TextRegion

MIN_CHUNK_CHARS = 80
MAX_CHUNK_CHARS = 900

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?!؟…])\s+|\n")


def _split_paragraph(text: str, max_chars: int) -> list[str]:
    """Split one paragraph into pieces <= max_chars at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    current = ""

    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue

        # Very long sentence without enders: split at commas, then hard-cut.
        while len(sentence) > max_chars:
            cut = sentence.rfind("،", 0, max_chars)
            if cut < max_chars // 2:
                cut = sentence.rfind(" ", 0, max_chars)
            if cut < max_chars // 2:
                cut = max_chars
            pieces.append(current + sentence[:cut].strip())
            current = ""
            sentence = sentence[cut:].strip()

        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        pieces.append(current)
    return pieces


def _pieces_to_chunks(
    pieces: list[str],
    *,
    document_id: str,
    chunk_type: ChunkType,
    section: str | None,
    section_path: list[str],
    page_start: int,
    page_end: int,
    bbox: list[float] | None,
    score: float,
    start_index: int,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """Pack pieces into chunks, merging tiny ones with the previous piece."""
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) < min_chars:
            combined = f"{merged[-1]} {piece}"
            if len(combined) <= max_chars:
                merged[-1] = combined
                continue
        merged.append(piece)

    result: list[Chunk] = []
    for offset, text in enumerate(merged):
        result.append(
            Chunk(
                chunk_id=f"{document_id}-c{start_index + offset:04d}",
                document_id=document_id,
                chunk_index=start_index + offset,
                chunk_type=chunk_type,
                text=text,
                page_start=page_start,
                page_end=page_end,
                section=section,
                section_path=list(section_path),
                bbox=bbox,
                score=score,
                char_count=len(text),
            )
        )
    return result


def chunk_regions(
    document_id: str,
    regions: list[TextRegion],
    *,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    merge_across_pages: bool = False,
    reset_sections_per_page: bool = True,
) -> list[Chunk]:
    """Chunk a flat list of regions (single page or multi-page stream).

    Headings start sections; body paragraphs accumulate; long paragraphs are
    split at sentence boundaries; tiny remnants merge forward.

    merge_across_pages=False flushes at every page boundary (the right
    default for slide-like pages where each page covers its own topic); set
    True for flowing textbooks so paragraphs cut by a page break reunite.
    reset_sections_per_page=True clears page-local diagram/caption sections
    at page starts so they cannot leak into later pages.
    """
    chunks: list[Chunk] = []
    section_path: list[str] = []
    current_section: str | None = None
    pending_heading: str | None = None
    pending_heading_page = 0
    pieces_buffer: list[str] = []
    buffer_page_start = 0
    buffer_page_end = 0
    buffer_bboxes: dict[int, list[float]] = {}
    buffer_scores: list[float] = []
    buffer_last_page = 0
    pending_special: list[str] | None = None
    pending_special_pages: list[int] = []
    pending_special_scores: list[float] = []
    pending_heading_score = 0.0

    def buffer_bbox() -> list[float] | None:
        if not buffer_bboxes:
            return None
        return [
            min(b[0] for b in buffer_bboxes.values()),
            min(b[1] for b in buffer_bboxes.values()),
            max(b[2] for b in buffer_bboxes.values()),
            max(b[3] for b in buffer_bboxes.values()),
        ]

    def emit_pending_specials() -> None:
        nonlocal pending_special, pending_special_pages, pending_special_scores
        if not pending_special:
            return
        chunks.extend(
            _pieces_to_chunks(
                [" ".join(pending_special)],
                document_id=document_id,
                chunk_type=ChunkType.SPECIAL,
                section=current_section,
                section_path=list(section_path),
                page_start=pending_special_pages[0],
                page_end=pending_special_pages[-1],
                bbox=None,
                score=(
                    sum(pending_special_scores) / len(pending_special_scores)
                    if pending_special_scores
                    else 0.0
                ),
                start_index=len(chunks),
                min_chars=min_chars,
                max_chars=max_chars,
            )
        )
        pending_special = None
        pending_special_pages = []
        pending_special_scores = []

    def flush_buffer() -> None:
        nonlocal pieces_buffer, buffer_page_start, buffer_page_end
        nonlocal buffer_bboxes, buffer_scores, pending_heading, pending_heading_page
        nonlocal pending_heading_score
        if not pieces_buffer:
            # A heading that never received body text is still content:
            # emit it as a heading chunk instead of dropping it. Pending
            # specials stay pending here so later neighbors can merge them.
            if pending_heading:
                chunks.extend(
                    _pieces_to_chunks(
                        [pending_heading],
                        document_id=document_id,
                        chunk_type=ChunkType.HEADING,
                        section=section_path[-1] if section_path else None,
                        section_path=list(section_path),
                        page_start=pending_heading_page or 1,
                        page_end=pending_heading_page or 1,
                        bbox=None,
                        score=pending_heading_score,
                        start_index=len(chunks),
                        min_chars=min_chars,
                        max_chars=max_chars,
                    )
                )
                pending_heading = None
                pending_heading_page = 0
                pending_heading_score = 0.0
            return

        text = " ".join(pieces_buffer).strip()
        chunk_type = (
            ChunkType.SPECIAL
            if all("\u0600" not in t and t.isascii() for t in pieces_buffer)
            else ChunkType.PARAGRAPH
        )
        section = current_section if current_section else pending_heading

        chunks.extend(
            _pieces_to_chunks(
                [text],
                document_id=document_id,
                chunk_type=chunk_type,
                section=section,
                section_path=list(section_path),
                page_start=buffer_page_start,
                page_end=buffer_page_end,
                bbox=buffer_bbox(),
                score=sum(buffer_scores) / len(buffer_scores) if buffer_scores else 0.0,
                start_index=len(chunks),
                min_chars=min_chars,
                max_chars=max_chars,
            )
        )
        emit_pending_specials()
        pieces_buffer = []
        buffer_bboxes = {}
        buffer_scores = []
        buffer_page_start = 0
        buffer_page_end = 0
        pending_heading = None
        pending_heading_page = 0
        pending_heading_score = 0.0

    def add_region_to_buffer(region: TextRegion) -> None:
        nonlocal buffer_page_start, buffer_page_end
        text = region.text.strip()
        if not text:
            return
        if not pieces_buffer:
            buffer_page_start = region.page_number
        buffer_page_end = max(buffer_page_end, region.page_number)
        pieces_buffer.append(text)
        union = [region.left, region.top, region.right, region.bottom]
        existing = buffer_bboxes.get(region.page_number)
        if existing is None:
            buffer_bboxes[region.page_number] = union
        else:
            buffer_bboxes[region.page_number] = [
                min(existing[0], union[0]),
                min(existing[1], union[1]),
                max(existing[2], union[2]),
                max(existing[3], union[3]),
            ]
        buffer_scores.append(region.mean_score)

    for region in regions:
        # Page boundary: flush and optionally reset page-local state.
        if region.page_number != buffer_last_page:
            first_transition = buffer_last_page != 0
            buffer_last_page = region.page_number
            if first_transition and not merge_across_pages:
                flush_buffer()
                if reset_sections_per_page:
                    section_path = []
                    current_section = None
                    pending_heading = None
                    pending_heading_page = 0

        if region.role == RegionRole.HEADING:
            flush_buffer()
            pending_heading = region.text
            pending_heading_page = region.page_number
            pending_heading_score = region.mean_score
            # Repeated headings (running titles across pages) never grow the
            # path; new ones cap the path at 3 levels.
            if region.text in section_path:
                section_path = section_path[: section_path.index(region.text) + 1]
            else:
                section_path = (section_path + [region.text])[-3:]
            current_section = region.text
            continue

        if region.role == RegionRole.SPECIAL:
            # Specials merge into the current chunk (charts, formulas read
            # better with their surrounding paragraph) but never start one.
            if pieces_buffer:
                projected = sum(len(p) for p in pieces_buffer) + len(region.text) + 1
                if projected <= max_chars:
                    add_region_to_buffer(region)
                    continue
            flush_buffer()
            if pieces_buffer:
                add_region_to_buffer(region)
                continue
            # Isolated special: hold as pending so scattered neighbors
            # (periodic-table cells, chart labels) merge instead of
            # shattering into tiny chunks. Note: flush_buffer above must not
            # emit pending specials before neighbors can join, so specials
            # are only emitted via emit_pending_specials on real flushes.
            if pending_special:
                pending_special.append(region.text)
                pending_special_pages.append(region.page_number)
                pending_special_scores.append(region.mean_score)
            else:
                pending_special = [region.text]
                pending_special_pages = [region.page_number]
                pending_special_scores = [region.mean_score]
            pending_heading = None
            continue

        # BODY region: flush when the accumulated text would exceed max_chars.
        projected = sum(len(p) for p in pieces_buffer) + len(region.text) + 1
        if pieces_buffer and projected > max_chars:
            flush_buffer()

        add_region_to_buffer(region)

        if sum(len(p) for p in pieces_buffer) >= max_chars:
            flush_buffer()

    # Tail: emit pending isolated specials, then the final buffer (which
    # also emits any pending heading with no body).
    emit_pending_specials()
    flush_buffer()

    # Renumber chunk_index sequentially.
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index
        chunk.chunk_id = f"{chunk.document_id}-c{index:04d}"

    return chunks


def chunk_document(document: DocumentResult) -> list[Chunk]:
    """Chunk a full DocumentResult: regions across all pages in reading order."""
    all_regions: list[TextRegion] = []
    for page in document.pages:
        all_regions.extend(page.regions)
    return chunk_regions(document.metadata.document_id, all_regions)


def build_chunks(document: DocumentResult) -> DocumentResult:
    """Attach chunks to the document (pipeline helper)."""
    document.chunks = chunk_document(document)
    return document
