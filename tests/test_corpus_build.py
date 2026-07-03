"""Tests for the deterministic corpus-building transforms."""

from __future__ import annotations

import pytest

from mizan.corpus_build import (
    build_documents,
    chunk_paragraphs,
    revision_url,
    split_paragraphs,
    title_to_slug,
)


def test_title_to_slug_replaces_spaces() -> None:
    assert title_to_slug("جامعة القرويين") == "جامعة_القرويين"
    assert title_to_slug("الرياض") == "الرياض"


def test_revision_url_is_a_pinned_permalink() -> None:
    assert revision_url(75752938) == "https://ar.wikipedia.org/w/index.php?oldid=75752938"


def test_split_paragraphs_collapses_whitespace_and_drops_blanks() -> None:
    extract = "  first   line \n\n\t \nsecond\tline  \n"
    assert split_paragraphs(extract) == ["first line", "second line"]


def test_split_paragraphs_empty() -> None:
    assert split_paragraphs("   \n\n \t ") == []


def test_chunk_paragraphs_packs_up_to_target() -> None:
    paragraphs = ["a" * 300, "b" * 300, "c" * 300]
    # 300 + 1 + 300 = 601 <= 700 fits; adding the third would exceed 700.
    chunks = chunk_paragraphs(paragraphs, target_chars=700, max_chars=1000)
    assert chunks == ["a" * 300 + " " + "b" * 300, "c" * 300]


def test_chunk_paragraphs_never_splits_a_paragraph_within_max() -> None:
    paragraphs = ["x" * 900]
    assert chunk_paragraphs(paragraphs, target_chars=700, max_chars=1000) == ["x" * 900]


def test_chunk_paragraphs_splits_oversized_paragraph_on_sentences() -> None:
    paragraph = "الجملة الأولى. الجملة الثانية. الجملة الثالثة."
    chunks = chunk_paragraphs([paragraph], target_chars=15, max_chars=15, min_chars=1)
    assert chunks == ["الجملة الأولى.", "الجملة الثانية.", "الجملة الثالثة."]
    assert all(len(c) <= 15 for c in chunks)


def test_chunk_paragraphs_hard_splits_a_sentence_with_no_punctuation() -> None:
    paragraph = " ".join(["كلمة"] * 20)  # one long "sentence", no terminal mark
    chunks = chunk_paragraphs([paragraph], target_chars=20, max_chars=20, min_chars=1)
    assert len(chunks) > 1
    assert all(len(c) <= 20 for c in chunks)


def test_chunk_paragraphs_merges_short_heading_into_next() -> None:
    # A lone heading (< min_chars) must not become its own document.
    chunks = chunk_paragraphs(["زحل", "z" * 600], target_chars=700, max_chars=1000, min_chars=50)
    assert chunks == ["زحل " + "z" * 600]


def test_chunk_paragraphs_merges_trailing_short_chunk_into_previous() -> None:
    chunks = chunk_paragraphs(["y" * 600, "خط"], target_chars=700, max_chars=1000, min_chars=50)
    assert chunks == ["y" * 600 + " خط"]


def test_chunk_paragraphs_is_deterministic() -> None:
    paragraphs = [f"paragraph number {i} with some filler text" for i in range(20)]
    first = chunk_paragraphs(paragraphs)
    second = chunk_paragraphs(paragraphs)
    assert first == second


def test_chunk_paragraphs_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        chunk_paragraphs(["a"], target_chars=0, max_chars=10)
    with pytest.raises(ValueError, match="min_chars <= target_chars <= max_chars"):
        chunk_paragraphs(["a"], target_chars=100, max_chars=10)


def test_build_documents_ids_titles_and_urls() -> None:
    extract = "المقطع الأول من النص.\n" + ("كلمة " * 200) + "\nالمقطع الأخير."
    docs = build_documents("الرياض", 75752938, extract, target_chars=300, max_chars=500)
    assert len(docs) >= 2
    assert [d.id for d in docs] == [f"arwiki:الرياض#{i}" for i in range(len(docs))]
    assert all(d.title == "الرياض" for d in docs)
    assert all(d.url == "https://ar.wikipedia.org/w/index.php?oldid=75752938" for d in docs)


def test_build_documents_rejects_empty_extract() -> None:
    with pytest.raises(ValueError, match="no text"):
        build_documents("فارغ", 1, "   \n\n  ")
