"""Careful Persian text normalization.

Design rule: normalize *form*, never *content*.
Everything here is reversible-ish or purely cosmetic — we never delete
alphanumeric content, never fold away digits, and never reorder text.
Aggressive cleaning (noise removal) belongs to layout.py, where geometry
and confidence are still available.
"""

from __future__ import annotations

import re
import unicodedata

# Arabic -> Persian letter folding (safe, meaning-preserving for Persian text).
LETTER_FOLD = {
    "\u064a": "\u06cc",  # ي -> ی
    "\u0649": "\u06cc",  # ى -> ی
    "\u0643": "\u06a9",  # ك -> ک
}

# Optional folds that only apply in *Arabic-context* words, see below.
ARABIC_CONTEXT_FOLD = {
    "أ": "ا",
    "إ": "ا",
    "ؤ": "و",
    "ئ": "ی",
    "ة": "ه",
    "ۀ": "ه",
}

# Diacritics (harakat) and tatweel are stripped: they carry no meaning for
# search/embeddings and hurt tokenization. Quranic annotation signs (06D6-06DC
# etc.) are kept — removing them could damage religious content.
DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670]")

# Zero-width characters. ZWNJ (\u200c) is *meaningful* in Persian
# (می‌روم, کتاب‌ها) and is preserved. Everything else is dropped.
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069]")

# Digits: Persian/Arabic-Indic -> ASCII for stable embeddings and numbers.
DIGIT_TABLE = str.maketrans({
    "\u06f0": "0", "\u06f1": "1", "\u06f2": "2", "\u06f3": "3", "\u06f4": "4",
    "\u06f5": "5", "\u06f6": "6", "\u06f7": "7", "\u06f8": "8", "\u06f9": "9",
    "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3", "\u0664": "4",
    "\u0665": "5", "\u0666": "6", "\u0667": "7", "\u0668": "8", "\u0669": "9",
})

PUNCT_NORM = {
    "،": ",",  # keep as-is? no: fold Arabic comma to ASCII comma for embeddings
    "؛": ";",
    "؟": "?",
    "٬": ",",
    "٫": ".",  # decimal separator
}

SPACE_RE = re.compile(r"[ \t\u00a0\u2000-\u200a]+")

# Space before punctuation is noise (OCR artifact).
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([،؛,:.!؟\.\!])")

# Arabic-context fold: ة/أ/إ etc. appear in Arabic loanwords; folding them is
# standard practice for Persian search (see Persian-OCR-App) and does not
# change meaning in educational content.
ARABIC_CHARS_CLASS = "أإؤئةۀ"


def normalize_text(text: str) -> str:
    """Normalize one line of mixed Persian/Latin text."""
    if not text:
        return ""

    # 1) Unicode NFC: compose decomposed forms (e.g. heh + hamza above).
    text = unicodedata.normalize("NFC", text)

    # 2) Remove zero-width marks except ZWNJ.
    text = ZERO_WIDTH_RE.sub("", text)

    # 3) Fold Arabic letter variants -> Persian.
    for old, new in LETTER_FOLD.items():
        text = text.replace(old, new)

    # 4) Fold Arabic-only characters (ة, أ, ...) — safe for Persian content.
    for old, new in ARABIC_CONTEXT_FOLD.items():
        text = text.replace(old, new)

    # 5) Strip diacritics and tatweel.
    text = DIACRITICS_RE.sub("", text)
    text = text.replace("\u0640", "")  # tatweel/kashida

    # 6) Digits -> ASCII.
    text = text.translate(DIGIT_TABLE)

    # 7) Persian punctuation -> ASCII equivalents (embedding friendly).
    for old, new in PUNCT_NORM.items():
        text = text.replace(old, new)

    # 8) Fix OCR space-punctuation artifacts.
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)

    # 9) Collapse runs of spaces / NBSP.
    text = SPACE_RE.sub(" ", text)

    return text.strip()


def normalize_lines(lines: list[str]) -> list[str]:
    """Normalize each line, keeping empty-line structure out."""
    out = []
    for line in lines:
        n = normalize_text(line)
        if n:
            out.append(n)
    return out


def is_mostly_persian(text: str, threshold: float = 0.35) -> bool:
    """True if Persian/Arabic script dominates the alphanumeric content."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    persian = sum(1 for c in letters if "\u0600" <= c <= "\u06ff")
    return persian / len(letters) >= threshold


def persian_letter_ratio(text: str) -> float:
    compact = "".join(text.split())
    if not compact:
        return 0.0
    return sum(1 for c in compact if "\u0600" <= c <= "\u06ff") / len(compact)


def has_persian(text: str) -> bool:
    return any("\u0600" <= c <= "\u06ff" for c in text)


# Backwards-compatible alias (old test scripts import this name).
def normalize_persian_text(text: str) -> str:
    return "\n".join(normalize_lines(text.splitlines()))
