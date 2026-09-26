"""Layout analysis.

Turns raw OCR blocks into structured page content:
    blocks -> lines (RTL order) -> columns (RTL reading order) -> regions
    (title/heading/body/special/noise) -> paragraphs, plus cross-page
    header/footer/watermark removal based on text repetition.
"""

from __future__ import annotations

import re
from collections import Counter

from models import LineType, OCRBlock, OCRLine, PageResult, RegionRole, TextRegion
from normalize import has_persian, persian_letter_ratio

MIN_BLOCK_SCORE = 0.55  # blocks below this are dropped outright
SPECIAL_MIN_SCORE = 0.30  # formula-like content is kept down to this score


# ---------------------------------------------------------------------------
# Content type helpers
# ---------------------------------------------------------------------------

PERSIAN_DIGIT_TRANS = str.maketrans({
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
})


def looks_like_formula(text: str) -> bool:
    """Chemical/math notation: CH4, H2O, NH3, E=mc2, 12Mg, 1γCl, a+b=c ..."""
    if not text:
        return False
    compact = text.replace(" ", "").translate(PERSIAN_DIGIT_TRANS)
    if len(compact) > 60:
        return False
    if re.fullmatch(r"[A-Za-z0-9+\-=→×·⇌()\[\]γδθλµ]{2,}", compact) and any(c.isdigit() for c in compact):
        return True
    if re.search(r"[A-Z][a-z]?\d", compact) and re.search(r"[+\-=→⇌]", compact):
        return True
    return False


def looks_like_english_word(text: str) -> bool:
    """Latin words like ACETIC, PHENOL, neutralization — real content."""
    if not text:
        return False
    compact = "".join(text.split())
    if not compact.isascii():
        return False
    letters = sum(c.isalpha() for c in compact)
    return letters >= 3 and letters / len(compact) >= 0.7


def _letter_spam_ratio(text: str) -> float:
    """Fraction of tokens that are 1-2 letter OCR debris tokens."""
    tokens = text.split()
    if len(tokens) < 3:
        return 0.0
    spam = sum(1 for token in tokens if len(token) <= 2)
    return spam / len(tokens)


_DECORATIVE_CHARS = set("0OoOo°○o")


def is_decorative_mark(text: str) -> bool:
    """Decorative page ornaments OCR'd as zeros/Os: "00 00", "O0", "۰۰"."""
    compact = "".join(text.split()).translate(PERSIAN_DIGIT_TRANS)
    return 0 < len(compact) <= 8 and all(c in _DECORATIVE_CHARS for c in compact)


def looks_like_chart_label(text: str) -> bool:
    """Short scattered data labels: "٪اکسیزن", "18٪ کرین", "یتاسیم 8", "21 Mg".

    These belong to diagrams/charts; they are preserved as SPECIAL content
    (merged with neighbors) but never become headings.
    """
    tokens = text.split()
    if len(tokens) > 3:
        return False
    if "٪" in text or "%" in text:
        return True
    for token in tokens:
        stripped = token.strip("().,")
        if stripped.isdigit():
            return True
        if len(stripped) <= 5 and stripped[0:1].isdigit():
            return True
    return False


def looks_like_footer_text(text: str) -> bool:
    """Banner/footer phrases from the source's own branding."""
    compact = "".join(text.split())
    markers = (
        "دانلوداز",
        "وبسایت",
        "وبسايت",
        "اپليکيشن",
        "اپلیکیشن",
        "اپلكيش",
        "پادرس",
        "@sadr_olom",
        "sadr_olom",
    )
    return any(marker in compact for marker in markers)


def is_noise_block(block: OCRBlock) -> bool:
    """Block-level noise filter. Never drops formula/word content."""
    text = block.text.strip()
    if not text:
        return True
    compact = "".join(text.split())

    # Footer/branding lines: dropped wherever they appear. Checked before
    # looks_like_english_word so "@sadr_olom" is not rescued as a Latin word.
    if looks_like_footer_text(text):
        return True

    if looks_like_formula(text) or looks_like_english_word(text):
        return False

    # Standalone Latin/symbol marks: decorative circles read as "O".
    if compact in {"O", "o", "0", "Q", "©", "®"} or is_decorative_mark(text):
        return True

    if len(compact) == 1 and block.score < 0.80:
        return True
    if len(compact) == 2 and block.score < 0.60:
        return True

    if not has_persian(text):
        # Short garbage fragments like "Juuttres", "Iw", "NM'".
        if len(compact) <= 8 and block.score < 0.75:
            return True
        return False

    ratio = persian_letter_ratio(text)
    if ratio < 0.45 and len(compact) <= 10 and block.score < 0.75:
        return True

    # Letter-spam from failed OCR on colored/decorated text.
    if _letter_spam_ratio(text) >= 0.5:
        return True

    return False


# ---------------------------------------------------------------------------
# Line grouping
# ---------------------------------------------------------------------------

def same_line(a: OCRBlock, b: OCRBlock) -> bool:
    overlap = min(a.bottom, b.bottom) - max(a.top, b.top)
    reference = min(a.height, b.height)
    return overlap >= 0.5 * reference


def group_into_lines(blocks: list[OCRBlock]) -> list[OCRLine]:
    """Group blocks into visual lines by vertical overlap, then order RTL."""
    if not blocks:
        return []

    blocks = sorted(blocks, key=lambda b: (b.top, b.left))
    lines: list[list[OCRBlock]] = []
    for block in blocks:
        best_line, best_overlap = None, 0.0
        for line in lines:
            anchor = line[-1]
            if same_line(anchor, block):
                overlap = min(anchor.bottom, block.bottom) - max(anchor.top, block.top)
                if overlap > best_overlap:
                    best_overlap, best_line = overlap, line
        if best_line is not None:
            best_line.append(block)
        else:
            lines.append([block])

    result: list[OCRLine] = []
    for line_blocks in lines:
        # RTL reading order: rightmost block first. Persian strings inside
        # blocks are never reversed.
        line_blocks.sort(key=lambda b: b.center_x, reverse=True)
        result.append(OCRLine(blocks=line_blocks, page_number=line_blocks[0].page_number))
    result.sort(key=lambda line: line.top)
    return result


# ---------------------------------------------------------------------------
# Columns + reading order
# ---------------------------------------------------------------------------

def detect_column_count(lines: list[OCRLine], page_width: float) -> int:
    """Detect 1 vs 2 text columns from line positions."""
    if len(lines) < 8:
        return 1
    midpoint = page_width / 2
    left = sum(1 for line in lines if line.right <= midpoint + 0.02 * page_width)
    right = sum(1 for line in lines if line.left >= midpoint - 0.02 * page_width)
    crossing = len(lines) - left - right
    if left >= 3 and right >= 3 and crossing <= max(2, int(len(lines) * 0.2)):
        return 2
    return 1


def order_lines_reading_order(lines: list[OCRLine], page_width: float) -> tuple[list[OCRLine], int]:
    """RTL reading order: right column top-to-bottom, then left column."""
    column_count = detect_column_count(lines, page_width)
    if column_count == 1:
        return sorted(lines, key=lambda line: line.top), 1

    midpoint = page_width / 2
    right_col = sorted((l for l in lines if l.center_x >= midpoint), key=lambda l: l.top)
    left_col = sorted((l for l in lines if l.center_x < midpoint), key=lambda l: l.top)
    return right_col + left_col, 2


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

LIST_MARKER_RE = re.compile(r"^\s*([0-9]{1,3}[-.)]|[a-zA-Z][-.)]|[-•●◦*])\s")


def _spam_line(line: OCRLine) -> bool:
    """A full line made of letter-spam tokens."""
    return _letter_spam_ratio(line.text) >= 0.5 and len(line.text.split()) >= 3


def classify_line(
    line: OCRLine,
    page_median_height: float | None = None,
    body_width_hint: float | None = None,
    page_height: float | None = None,
) -> LineType:
    """Classify a line as BODY / HEADING / SPECIAL / NOISE."""
    text = line.text.strip()
    if not text:
        return LineType.NOISE

    # Bottom-of-page banners (site/branding strips) are never content.
    if page_height is not None and line.bottom >= page_height * 0.90:
        if looks_like_footer_text(text) or len(text.split()) <= 6:
            return LineType.NOISE

    if looks_like_footer_text(text):
        return LineType.NOISE

    if is_decorative_mark(text):
        return LineType.NOISE

    if _spam_line(line):
        return LineType.NOISE

    compact = "".join(text.split())

    if looks_like_formula(text) or looks_like_chart_label(text):
        return LineType.SPECIAL
    if looks_like_english_word(text) and not has_persian(text):
        return LineType.SPECIAL

    if not has_persian(text):
        if len(compact) <= 3:
            return LineType.NOISE
        if len(compact) <= 10 and line.mean_score < 0.75:
            return LineType.NOISE
        if sum(1 for c in compact if c.isalpha()) < 3:
            return LineType.NOISE

    ratio = persian_letter_ratio(text)
    if ratio < 0.35 and len(compact) <= 12 and line.mean_score < 0.75:
        return LineType.NOISE

    # --- heading heuristics ---
    # A short line *ending with a light verb* is the tail of a broken-up
    # sentence, not a title: "... توليد هيدروكلريک / اسيد كاربرد دارد".
    LIGHT_VERB_ENDINGS = (
        "مى دهد", "می دهد", "مي دهد", "می‌دهد",
        "می شود", "مى شود", "مي شود", "می‌شود", "میشود",
        "می شوند", "مى شوند", "مي شوند", "می‌شوند", "میشوند",
        "می باشند", "مى باشند", "میباشد", "می باشد", "ميباشد",
        "می گردد", "میگردد", "می گیرد", "میگیرد", "می آید", "میآید",
        "می رود", "میرود", "می رود",
        "دارد", "دارند", "است", "بود", "باشد", "شد", "شده است",
    )
    if text.rstrip().endswith(LIGHT_VERB_ENDINGS):
        return LineType.BODY

    # Numbered/figure captions inside the body flow.
    word_count = len(text.split())

    # Sentence tails and captions are not headings. Light verbs at either
    # end indicate a broken-up body sentence.
    _light_starts = (
        "دارند", "دارد", "است", "باشد", "بود", "شده", "می", "مي", "مى",
        "شود", "شوند", "دهند", "کند", "کنند",
    )
    if text.split()[0].startswith(_light_starts):
        return LineType.BODY

    # Lines containing stray 1-2 char digit tokens ("زیاد 4 مقاومت") are
    # bullet/list merges from the layout, never titles.
    if any(token.strip("().,") .isdigit() and len(token.strip("().,")) <= 2 for token in text.split()):
        return LineType.BODY

    # Real headings have at least two words; single-word diagram labels
    # ("کمبر", "خالص") must not hijack the section path.
    if 2 <= word_count <= 8:
        short_relative_to_page = (
            line.width < 0.5 * body_width_hint if body_width_hint else line.width < 300
        )
        taller_than_body = (
            page_median_height is not None
            and line.height > page_median_height * 1.25
        )
        ends_without_punct = not text.rstrip().endswith((",", ".", "،", ";", ":"))
        if short_relative_to_page and (taller_than_body or word_count <= 6) and ends_without_punct:
            return LineType.HEADING

    return LineType.BODY


# ---------------------------------------------------------------------------
# Regions (paragraphs, headings, special blocks)
# ---------------------------------------------------------------------------

def line_gap(a: OCRLine, b: OCRLine) -> float:
    return b.top - a.bottom


def build_regions(
    lines: list[OCRLine],
    page_median_height: float | None = None,
    page_height: float | None = None,
) -> list[TextRegion]:
    """Group ordered lines into regions: heading, body paragraphs, specials."""
    if not lines:
        return []

    if page_median_height is None:
        page_median_height = statistics_median([line.height for line in lines])

    regions: list[TextRegion] = []
    current_body: list[OCRLine] = []
    current_heading: OCRLine | None = None

    def flush_body() -> None:
        nonlocal current_body
        if current_body:
            regions.append(
                TextRegion(
                    role=RegionRole.BODY,
                    lines=current_body,
                    page_number=current_body[0].page_number,
                )
            )
            current_body = []

    def flush_heading() -> None:
        nonlocal current_heading
        if current_heading is not None:
            regions.append(
                TextRegion(
                    role=RegionRole.HEADING,
                    lines=[current_heading],
                    page_number=current_heading.page_number,
                )
            )
            current_heading = None

    current_special: list[OCRLine] = []

    def flush_special() -> None:
        nonlocal current_special
        if current_special:
            regions.append(
                TextRegion(
                    role=RegionRole.SPECIAL,
                    lines=current_special,
                    page_number=current_special[0].page_number,
                )
            )
            current_special = []

    for line in lines:
        line_type = classify_line(
            line,
            page_median_height,
            body_width_hint=page_width_hint(lines),
            page_height=page_height,
        )

        if line_type == LineType.NOISE:
            flush_special()
            flush_heading()
            flush_body()
            continue

        if line_type == LineType.HEADING:
            flush_special()
            flush_body()
            flush_heading()
            current_heading = line
            continue

        if line_type == LineType.SPECIAL:
            flush_body()
            flush_heading()
            # Adjacent special lines merge into one region so poster
            # fragments become one chunk, not six.
            if current_special and line_gap(current_special[-1], line) > 3.0 * page_median_height:
                flush_special()
            current_special.append(line)
            continue

        flush_special()

        # BODY line: start new paragraph after a big vertical gap or after
        # an indented first line (typical Persian paragraph indent).
        if current_body:
            previous = current_body[-1]
            gap = line_gap(previous, line)
            big_gap = gap > 0.9 * page_median_height
            if big_gap:
                flush_body()

        if current_heading is not None:
            # Body directly after a heading starts a new paragraph.
            if current_body:
                flush_body()
            current_heading = None

        current_body.append(line)

    flush_special()
    flush_body()
    flush_heading()
    return regions


def page_width_hint(lines: list[OCRLine]) -> float | None:
    if not lines:
        return None
    return max(line.width for line in lines)


def statistics_median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    if n % 2 == 1:
        return values[n // 2]
    return (values[n // 2 - 1] + values[n // 2]) / 2


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

def annotate_headings(regions: list[TextRegion]) -> None:
    """Attach the nearest preceding heading to each region."""
    current_heading: str | None = None
    for region in regions:
        if region.role == RegionRole.HEADING:
            current_heading = region.text
        elif region.role == RegionRole.BODY:
            region.heading_text = current_heading


def build_page_result(
    page_number: int,
    width: float,
    height: float,
    blocks: list[OCRBlock],
    debug_keep_noise: bool = False,
) -> PageResult:
    """Assemble one page: filter -> lines -> reading order -> regions."""
    kept_blocks: list[OCRBlock] = []
    dropped_blocks: list[OCRBlock] = []
    for block in blocks:
        if block.score < MIN_BLOCK_SCORE and not (
            looks_like_formula(block.text) and block.score >= SPECIAL_MIN_SCORE
        ):
            dropped_blocks.append(block)
            continue
        if is_noise_block(block):
            dropped_blocks.append(block)
            continue
        kept_blocks.append(block)

    lines = group_into_lines(kept_blocks)
    ordered_lines, column_count = order_lines_reading_order(lines, width)

    regions = build_regions(ordered_lines, page_height=height)
    annotate_headings(regions)

    # Column index per region (0 = single column).
    for region in regions:
        region.column = 0

    page = PageResult(
        page_number=page_number,
        width=width,
        height=height,
        blocks=kept_blocks if not debug_keep_noise else blocks,
        lines=ordered_lines,
        regions=regions,
        column_count=column_count,
        headings=[r.text for r in regions if r.role in (RegionRole.HEADING, RegionRole.TITLE)],
        text="\n\n".join(r.text for r in regions if r.text),
    )
    return page


# ---------------------------------------------------------------------------
# Cross-page repeated content (headers / footers / watermarks)
# ---------------------------------------------------------------------------

def find_repeated_texts(pages: list[PageResult], min_ratio: float = 0.6) -> set[str]:
    """Text lines that appear (nearly) identically on >= min_ratio of pages."""
    if len(pages) < 4:
        return set()

    counts: Counter = Counter()
    for page in pages:
        seen = {line.text.strip() for line in page.lines if len(line.text.strip()) >= 3}
        for text in seen:
            counts[text] += 1

    threshold = max(2, int(len(pages) * min_ratio))
    return {text for text, count in counts.items() if count >= threshold}


def strip_repeated_content(pages: list[PageResult]) -> set[str]:
    """Remove header/footer/watermark lines repeated across pages."""
    repeated = find_repeated_texts(pages)
    if not repeated:
        return repeated

    for page in pages:
        page.lines = [line for line in page.lines if line.text.strip() not in repeated]
        for region in page.regions:
            region.lines = [line for line in region.lines if line.text.strip() not in repeated]
        page.regions = [r for r in page.regions if r.lines]
        page.text = "\n\n".join(r.text for r in page.regions if r.text)
        page.headings = [r.text for r in page.regions if r.role in (RegionRole.HEADING, RegionRole.TITLE)]
    return repeated


# Backwards-compatible names used by the old test scripts.
def extract_blocks(result, page_height: float | None = None) -> list[OCRBlock]:
    """Convert a raw RapidOCR result object to OCRBlocks (legacy helper)."""
    blocks: list[OCRBlock] = []
    if result is None or result.boxes is None:
        return blocks
    for box, text, score in zip(result.boxes, result.txts, result.scores):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        blocks.append(
            OCRBlock(
                text=str(text).strip(),
                score=float(score),
                left=min(xs),
                top=min(ys),
                right=max(xs),
                bottom=max(ys),
            )
        )
    return blocks


def group_into_paragraphs(lines: list[OCRLine]) -> list[list[OCRLine]]:
    """Legacy: group lines into paragraph lists."""
    regions = build_regions(lines)
    return [region.lines for region in regions]


def paragraphs_to_text(paragraphs: list[list[OCRLine]]) -> str:
    """Legacy: render paragraph groups as text."""
    parts = []
    for paragraph in paragraphs:
        text = " ".join(line.text for line in paragraph).strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)
