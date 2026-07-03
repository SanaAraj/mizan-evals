"""Arabic text normalization and tokenization for lexical retrieval.

Normalization folds orthographic variation that should not affect matching:
diacritics (tashkeel) and tatweel are stripped, and alef/ya/ta-marbuta/hamza
carriers are unified. This mirrors standard Arabic IR preprocessing so that, for
example, ``الرِّياض`` and ``الرياض`` tokenize identically.
"""

from __future__ import annotations

import re

# Codepoints removed outright: tatweel, superscript alef, and the tashkeel block
# (tanwin, harakat, shadda, sukun, maddah, and hamza-above/below marks).
_TABLE: dict[int, str | None] = {0x0640: None, 0x0670: None}
for _cp in range(0x064B, 0x0656):
    _TABLE[_cp] = None

# Carrier unification.
_TABLE.update(
    {
        ord(src): dst
        for src, dst in {
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ٱ": "ا",
            "ى": "ي",
            "ئ": "ي",
            "ؤ": "و",
            "ة": "ه",
        }.items()
    }
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize_arabic(text: str) -> str:
    """Return ``text`` with Arabic diacritics stripped and carriers unified."""
    return text.translate(_TABLE).lower()


def tokenize(text: str) -> list[str]:
    """Normalize then split ``text`` into word tokens (Arabic, Latin, digits)."""
    return _TOKEN_RE.findall(normalize_arabic(text))
