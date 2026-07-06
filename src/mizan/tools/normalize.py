"""Argument normalization: score a localized argument value as correct.

Gold arguments are stored in one canonical English form (``"Riyadh"``, ``"SAR"``,
``"tomorrow"``). A model answering an Arabic query will often emit the *localized*
surface form instead (``"الرياض"``, ``"ريال"``, ``"بكرة"``). Treating that as
wrong would confound two different things — the model picking the wrong value, and
the model simply not translating the value into English. This layer resolves a
produced value through per-category alias tables so a localized-but-correct value
scores as a match, and separately reports *whether* an alias was needed, which is
what the per-language ``localization_rate`` metric is built on.

The design decision behind that metric (localization is a finding, not an error)
is Sana's; see the dataset README and the scorer docstring.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

# Arabic-Indic digits → ASCII, so "٣" compares equal to "3".
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
# Alef variants unified to bare alef so "أبوظبي" and "ابوظبي" collapse together.
_ALEF_VARIANTS = str.maketrans("أإآ", "ااا")
# Tashkeel (harakat) and tatweel are stripped before comparison.
_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟـ]")


def _canon(value: object) -> str:
    """Normalize any value to a canonical comparison string.

    Applies digit folding, Arabic diacritic/tatweel removal, alef unification,
    whitespace collapsing, and Unicode-aware case folding. Latin text is
    lowercased; Arabic letters are left intact apart from the folds above.
    """
    text = str(value).strip().translate(_ARABIC_INDIC_DIGITS)
    text = _ARABIC_DIACRITICS.sub("", text).translate(_ALEF_VARIANTS)
    return re.sub(r"\s+", " ", text).casefold()


# Localized surface form → canonical English value, grouped by category purely for
# readability. Every canonical value is written the way gold arguments are (its
# own ``_canon`` is what actually gets compared). Categories with a genuine
# surface-form collision across values (bare "كيلو" for both kilometre and
# kilogram) are left out; the dataset uses the unambiguous forms below.
_ALIAS_GROUPS: dict[str, dict[str, str]] = {
    "cities": {
        "الرياض": "Riyadh",
        "جدة": "Jeddah",
        "مكة": "Mecca",
        "مكة المكرمة": "Mecca",
        "المدينة": "Medina",
        "المدينة المنورة": "Medina",
        "الدمام": "Dammam",
        "الخبر": "Khobar",
        "دبي": "Dubai",
        "أبوظبي": "Abu Dhabi",
        "الشارقة": "Sharjah",
        "الدوحة": "Doha",
        "الكويت": "Kuwait City",
        "المنامة": "Manama",
        "مسقط": "Muscat",
        "القاهرة": "Cairo",
        "الإسكندرية": "Alexandria",
        "عمّان": "Amman",
        "بيروت": "Beirut",
        "بغداد": "Baghdad",
        "لندن": "London",
        "باريس": "Paris",
        "نيويورك": "New York",
        "إسطنبول": "Istanbul",
    },
    "currencies": {
        "ريال": "SAR",
        "ريال سعودي": "SAR",
        "درهم": "AED",
        "درهم إماراتي": "AED",
        "دولار": "USD",
        "دولار أمريكي": "USD",
        "يورو": "EUR",
        "جنيه إسترليني": "GBP",
        "جنيه": "EGP",
        "جنيه مصري": "EGP",
        "دينار كويتي": "KWD",
        "ريال قطري": "QAR",
        "ين": "JPY",
        "ين ياباني": "JPY",
    },
    "languages": {
        "الإنجليزية": "English",
        "الانجليزية": "English",
        "إنجليزي": "English",
        "انجليزي": "English",
        "العربية": "Arabic",
        "عربي": "Arabic",
        "الفرنسية": "French",
        "فرنسي": "French",
        "الإسبانية": "Spanish",
        "الألمانية": "German",
        "التركية": "Turkish",
        "الصينية": "Chinese",
        "اليابانية": "Japanese",
        "الأردية": "Urdu",
    },
    "units": {
        "كيلومتر": "kilometer",
        "متر": "meter",
        "ميل": "mile",
        "قدم": "foot",
        "كيلوغرام": "kilogram",
        "كيلوجرام": "kilogram",
        "غرام": "gram",
        "جرام": "gram",
        "رطل": "pound",
        "باوند": "pound",
        "أونصة": "ounce",
        "لتر": "liter",
        "غالون": "gallon",
        "جالون": "gallon",
        "مئوية": "celsius",
        "درجة مئوية": "celsius",
        "فهرنهايت": "fahrenheit",
    },
    "dates": {
        "اليوم": "today",
        "غدا": "tomorrow",
        "بكرة": "tomorrow",
        "باچر": "tomorrow",
        "أمس": "yesterday",
        "بعد غد": "day after tomorrow",
        "بعد بكرة": "day after tomorrow",
        "الليلة": "tonight",
    },
    "travel_modes": {
        "بالسيارة": "driving",
        "سيارة": "driving",
        "مشي": "walking",
        "على الأقدام": "walking",
        "مواصلات": "transit",
        "نقل عام": "transit",
        "دراجة": "cycling",
        "بالدراجة": "cycling",
    },
}

# Flattened, fully-canonicalized alias table: canon(localized) -> canon(English).
_ALIASES: dict[str, str] = {
    _canon(surface): _canon(canonical)
    for group in _ALIAS_GROUPS.values()
    for surface, canonical in group.items()
}
# The set of canonical values that *have* a localized alias — i.e. the arguments
# for which "localization" is even meaningful (the denominator of the rate).
_LOCALIZABLE: frozenset[str] = frozenset(_ALIASES.values())


class ResolvedValue(BaseModel):
    """A produced argument value after alias resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical: str
    localized: bool


def resolve_value(value: object) -> ResolvedValue:
    """Resolve a produced value to canonical form, flagging alias use.

    ``localized`` is ``True`` when ``value`` was a recognized localized surface
    form that had to be mapped through the alias table (e.g. ``"الرياض"`` →
    ``"riyadh"``), and ``False`` when it was already canonical.
    """
    canon = _canon(value)
    aliased = _ALIASES.get(canon)
    if aliased is not None:
        return ResolvedValue(canonical=aliased, localized=True)
    return ResolvedValue(canonical=canon, localized=False)


def _numeric_equal(a: object, b: object) -> bool:
    """Return whether two values are equal as numbers (e.g. ``4`` and ``4.0``)."""
    try:
        return float(_canon(a)) == float(_canon(b))
    except ValueError:
        return False


class ArgumentMatch(BaseModel):
    """The outcome of comparing one produced argument value against gold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    matched: bool
    localizable: bool
    localized: bool


def match_argument(gold_value: object, produced_value: object) -> ArgumentMatch:
    """Compare a produced argument value against its gold canonical value.

    A value matches if it resolves to the same canonical form as gold (after
    alias resolution) or is numerically equal. ``localizable`` says the gold
    value has a localized alias at all; ``localized`` says the model produced that
    localized form for a value that matched — the signal the localization rate is
    computed from.
    """
    gold = resolve_value(gold_value)
    produced = resolve_value(produced_value)
    matched = gold.canonical == produced.canonical or _numeric_equal(gold_value, produced_value)
    localizable = gold.canonical in _LOCALIZABLE
    return ArgumentMatch(
        matched=matched,
        localizable=localizable,
        localized=matched and localizable and produced.localized,
    )


def is_localizable(gold_value: object) -> bool:
    """Return whether a gold value has any localized alias form."""
    return resolve_value(gold_value).canonical in _LOCALIZABLE
