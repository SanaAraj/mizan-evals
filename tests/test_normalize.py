"""Tests for Arabic normalization and tokenization."""

from __future__ import annotations

from mizan.retrieval.normalize import normalize_arabic, tokenize


def test_strips_diacritics() -> None:
    # "al-Riyāḍ" written with shadda/kasra/fatha collapses to the bare form.
    assert normalize_arabic("الرِّيَاض") == "الرياض"


def test_removes_tatweel() -> None:
    assert normalize_arabic("الـــرياض") == "الرياض"


def test_unifies_alef_variants() -> None:
    assert normalize_arabic("أحمد") == "احمد"
    assert normalize_arabic("إسلام") == "اسلام"
    assert normalize_arabic("آية") == "ايه"


def test_unifies_ya_and_ta_marbuta() -> None:
    assert normalize_arabic("على") == "علي"
    assert normalize_arabic("مكتبة") == "مكتبه"


def test_lowercases_latin() -> None:
    assert normalize_arabic("Riyadh") == "riyadh"


def test_tokenize_splits_on_punctuation() -> None:
    assert tokenize("مرحبا، كيف حالك؟") == ["مرحبا", "كيف", "حالك"]


def test_tokenize_normalizes_before_splitting() -> None:
    # Diacritics must not create distinct tokens.
    assert tokenize("الرِّياض") == tokenize("الرياض")
