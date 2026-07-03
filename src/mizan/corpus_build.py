"""Deterministic transforms for building a retrieval corpus from Wikipedia.

The corpus is assembled from pinned Arabic Wikipedia revisions. Everything in
this module is pure and side-effect free so it can be unit-tested without a
network: given the plaintext extract of a revision, it produces a stable list of
:class:`~mizan.corpus.Document` chunks. The network fetch and the revision-pin
check live in ``scripts/build_corpus.py``.

Chunking is greedy paragraph packing: paragraphs are concatenated up to a target
size, never splitting a paragraph unless it alone exceeds the hard maximum (in
which case it is split on sentence boundaries). The same extract therefore always
yields the same chunks with the same ids, which is what makes a committed corpus
reproducible.
"""

from __future__ import annotations

import re

from mizan.corpus import Document

# Whitespace inside a line is collapsed; paragraph breaks (newlines) are kept.
_INLINE_WS_RE = re.compile(r"[ \t ]+")
# Sentence break: after an Arabic/Latin terminal mark followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.؟!])\s+")

#: Canonical revision-permalink form; ``oldid`` alone uniquely identifies text.
REVISION_PERMALINK = "https://ar.wikipedia.org/w/index.php?oldid={revid}"


def title_to_slug(title: str) -> str:
    """Turn an article title into the id slug used in ``arwiki:<slug>#<n>``.

    Spaces become underscores to match the ``arwiki:الرياض#0`` id convention
    already used by the evaluation items' gold labels.
    """
    return title.replace(" ", "_")


def revision_url(revid: int) -> str:
    """Return the stable Wikipedia permalink for a specific revision id."""
    return REVISION_PERMALINK.format(revid=revid)


def split_paragraphs(extract: str) -> list[str]:
    """Split a plaintext extract into cleaned, non-empty paragraphs.

    Inline runs of spaces/tabs are collapsed to a single space; blank lines are
    dropped. Order is preserved.
    """
    paragraphs: list[str] = []
    for raw_line in extract.split("\n"):
        line = _INLINE_WS_RE.sub(" ", raw_line).strip()
        if line:
            paragraphs.append(line)
    return paragraphs


def _pack(pieces: list[str], limit: int) -> list[str]:
    """Greedily concatenate ``pieces`` (space-joined) up to ``limit`` chars."""
    packed: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > limit:
            packed.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        packed.append(current)
    return packed


def _split_oversized(paragraph: str, max_chars: int) -> list[str]:
    """Break a paragraph longer than ``max_chars`` into pieces within the limit.

    Sentences are grouped first; a single sentence that still exceeds the limit
    (rare — no terminal punctuation) is hard-split on word boundaries. Every
    returned piece is ``<= max_chars`` as long as no single word exceeds it.
    """
    sentences: list[str] = []
    for sentence in (s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip()):
        if len(sentence) > max_chars:
            sentences.extend(_pack(sentence.split(" "), max_chars))
        else:
            sentences.append(sentence)
    return _pack(sentences, max_chars)


def _merge_short(chunks: list[str], min_chars: int) -> list[str]:
    """Fold chunks shorter than ``min_chars`` into a neighbour.

    A short chunk (typically a lone section heading) is prepended to the next
    chunk; a trailing short chunk is appended to the previous one. This keeps the
    corpus free of stray one-word documents.
    """
    merged: list[str] = []
    pending = ""
    for chunk in chunks:
        if pending:
            chunk = f"{pending} {chunk}".strip()
            pending = ""
        if len(chunk) < min_chars:
            pending = chunk
        else:
            merged.append(chunk)
    if pending:
        if merged:
            merged[-1] = f"{merged[-1]} {pending}".strip()
        else:
            merged.append(pending)
    return merged


def chunk_paragraphs(
    paragraphs: list[str],
    *,
    target_chars: int = 700,
    max_chars: int = 1000,
    min_chars: int = 50,
) -> list[str]:
    """Pack paragraphs into chunks of roughly ``target_chars``.

    A paragraph is never split unless it alone exceeds ``max_chars`` (then it is
    broken on sentence, and if needed word, boundaries). Chunks shorter than
    ``min_chars`` are merged into a neighbour so stray headings do not become
    their own documents. Packing is deterministic: the same input always yields
    the same chunks, which is what makes a committed corpus reproducible.

    Raises:
        ValueError: if the size bounds are non-positive or inconsistent
            (``min_chars <= target_chars <= max_chars`` must hold).
    """
    if min_chars <= 0 or target_chars <= 0 or max_chars <= 0:
        raise ValueError("size bounds must be positive")
    if not min_chars <= target_chars <= max_chars:
        raise ValueError("bounds must satisfy min_chars <= target_chars <= max_chars")

    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
        else:
            units.extend(_split_oversized(paragraph, max_chars))

    return _merge_short(_pack(units, target_chars), min_chars)


def build_documents(
    title: str,
    revid: int,
    extract: str,
    *,
    target_chars: int = 700,
    max_chars: int = 1000,
    min_chars: int = 50,
) -> list[Document]:
    """Turn one revision's plaintext extract into ordered corpus documents.

    Document ids follow ``arwiki:<title-slug>#<chunk-index>``; every chunk of an
    article shares the article title and a revision-pinned source ``url``.

    Raises:
        ValueError: if the extract yields no non-empty text.
    """
    chunks = chunk_paragraphs(
        split_paragraphs(extract),
        target_chars=target_chars,
        max_chars=max_chars,
        min_chars=min_chars,
    )
    if not chunks:
        raise ValueError(f"extract for {title!r} produced no text")
    slug = title_to_slug(title)
    url = revision_url(revid)
    return [
        Document(id=f"arwiki:{slug}#{index}", title=title, text=chunk, url=url)
        for index, chunk in enumerate(chunks)
    ]
