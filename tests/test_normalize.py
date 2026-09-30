"""Minimal tests for conservative Persian normalization.

Pins the guarantees the rest of the engine relies on:
form changes, content does not; Persian strings are never reversed;
ZWNJ survives; Arabic folds to Persian; digits fold to ASCII.
"""

from __future__ import annotations

from darsio_ingest.normalize import (
    has_persian,
    is_mostly_persian,
    normalize_text,
    persian_letter_ratio,
)


class TestLetterFolding:
    def test_arabic_yeh_to_persian_yeh(self):
        assert normalize_text("كتاب") == "کتاب"
        assert normalize_text("ي") == "ی"

    def test_arabic_kaf_to_persian_kaf(self):
        assert normalize_text("يك") == "یک"

    def test_arabic_context_folds(self):
        assert normalize_text("ؤ") == "و"
        assert normalize_text("ة") == "ه"
        assert normalize_text("أ") == "ا"

    def test_digit_folding_to_ascii(self):
        assert normalize_text("۱۲۳") == "123"
        assert normalize_text("٠١٢") == "012"
        assert normalize_text("۵/۲") == "5/2"


class TestPreservation:
    def test_zwnj_is_preserved(self):
        text = "می‌روم"
        assert normalize_text(text) == text

    def test_no_reversal_of_persian_strings(self):
        text = "شیمی فیزیک"
        result = normalize_text(text)
        assert result == text  # unchanged; especially not reversed
        assert result.index("شیمی") < result.index("فیزیک")

    def test_no_content_deletion(self):
        text = "آب و هوا در ۱۲ شهر"
        result = normalize_text(text)
        for token in ("آب", "هوا", "12", "شهر"):
            assert token.replace("۱۲", "12") in result

    def test_spaces_collapse_but_words_survive(self):
        result = normalize_text("سلام    دنیا")
        assert result == "سلام دنیا"

    def test_diacritics_stripped_but_letters_kept(self):
        result = normalize_text("کِتَاب")
        assert result == "کتاب"


class TestPunctuation:
    def test_persian_comma_folds_to_ascii(self):
        assert normalize_text("نیک، خوب") == "نیک, خوب"

    def test_space_before_punct_removed(self):
        assert normalize_text("سلام !") == "سلام!"


class TestClassifiers:
    def test_has_persian(self):
        assert has_persian("سلام")
        assert not has_persian("hello")

    def test_is_mostly_persian(self):
        assert is_mostly_persian("این یک آزمایش است")
        assert not is_mostly_persian("hello world")

    def test_persian_letter_ratio_bounds(self):
        assert persian_letter_ratio("سلام دنیا") > 0.9
        assert persian_letter_ratio("hello") == 0.0
        assert persian_letter_ratio("") == 0.0
