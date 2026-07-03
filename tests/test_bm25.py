"""Tests for the BM25 retriever."""

from __future__ import annotations

import pytest

from mizan.corpus import Document
from mizan.retrieval.bm25 import BM25Retriever

_CORPUS = [
    Document(id="d-riyadh", title="الرياض", text="الرياض هي عاصمة المملكة العربية السعودية."),
    Document(id="d-cairo", title="القاهرة", text="القاهرة هي عاصمة جمهورية مصر العربية."),
    Document(id="d-doha", title="الدوحة", text="الدوحة هي عاصمة دولة قطر."),
]


def test_retrieves_relevant_document_first() -> None:
    retriever = BM25Retriever(_CORPUS)
    ranked = retriever.retrieve("ما هي عاصمة السعودية؟", k=3)
    assert ranked[0] == "d-riyadh"


def test_retrieval_is_diacritic_insensitive() -> None:
    retriever = BM25Retriever(_CORPUS)
    with_marks = retriever.retrieve("عاصِمة السُّعوديّة", k=1)
    without = retriever.retrieve("عاصمة السعودية", k=1)
    assert with_marks == without == ["d-riyadh"]


def test_k_limits_results() -> None:
    retriever = BM25Retriever(_CORPUS)
    assert len(retriever.retrieve("عاصمة", k=2)) == 2


def test_zero_k_raises() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        BM25Retriever(_CORPUS).retrieve("q", k=0)


def test_empty_corpus_raises() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        BM25Retriever([])


def test_ranking_is_deterministic() -> None:
    retriever = BM25Retriever(_CORPUS)
    assert retriever.retrieve("عاصمة", k=3) == retriever.retrieve("عاصمة", k=3)
