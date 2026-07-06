"""Tests for the argument normalization / alias layer."""

from __future__ import annotations

import pytest

from mizan.tools.normalize import is_localizable, match_argument, resolve_value


def test_english_gold_resolves_without_localization() -> None:
    resolved = resolve_value("Riyadh")
    assert resolved.canonical == "riyadh"
    assert resolved.localized is False


def test_arabic_city_resolves_to_canonical_and_flags_localized() -> None:
    resolved = resolve_value("الرياض")
    assert resolved.canonical == "riyadh"
    assert resolved.localized is True


def test_match_localized_city_counts_as_correct_and_localized() -> None:
    match = match_argument("Riyadh", "الرياض")
    assert match.matched
    assert match.localizable
    assert match.localized


def test_match_english_value_correct_but_not_localized() -> None:
    match = match_argument("Riyadh", "Riyadh")
    assert match.matched
    assert match.localizable  # a city has a localized form...
    assert not match.localized  # ...but the model did not use it


def test_case_and_whitespace_insensitive() -> None:
    assert match_argument("New York", "  new   york ").matched


def test_alef_variant_and_diacritics_normalized() -> None:
    # "أبوظبي" (with hamza) vs the bare-alef canonical alias key.
    assert match_argument("Abu Dhabi", "ابوظبي").matched


def test_arabic_indic_digits_match_ascii() -> None:
    assert match_argument("3", "٣").matched


def test_numeric_equality_int_vs_float() -> None:
    assert match_argument(4, 4.0).matched
    assert match_argument(15, "15").matched


def test_currency_alias() -> None:
    match = match_argument("SAR", "ريال")
    assert match.matched and match.localized


def test_relative_date_alias_gulf() -> None:
    # "باچر" is Gulf for tomorrow.
    match = match_argument("tomorrow", "باچر")
    assert match.matched and match.localized


def test_travel_mode_alias() -> None:
    assert match_argument("driving", "بالسيارة").matched


def test_wrong_value_does_not_match() -> None:
    match = match_argument("Riyadh", "Jeddah")
    assert not match.matched
    assert not match.localized


def test_non_localizable_value_flags_false() -> None:
    # A free-text argument (a reminder task) has no alias table.
    assert not is_localizable("call the dentist")
    match = match_argument("call the dentist", "call the dentist")
    assert match.matched and not match.localizable and not match.localized


@pytest.mark.parametrize(
    ("gold", "produced"),
    [("Mecca", "مكة المكرمة"), ("Medina", "المدينة المنورة"), ("English", "الإنجليزية")],
)
def test_multiword_and_article_aliases(gold: str, produced: str) -> None:
    assert match_argument(gold, produced).matched
