"""Construct a retriever from a :class:`RetrieverSpec`."""

from __future__ import annotations

from collections.abc import Sequence

from mizan.config import RetrieverSpec
from mizan.corpus import Document
from mizan.retrieval.bm25 import BM25Retriever


def build_retriever(spec: RetrieverSpec, documents: Sequence[Document]) -> BM25Retriever:
    """Build the retriever named by ``spec.type`` over ``documents``.

    Raises:
        ValueError: for an unknown retriever type.
    """
    if spec.type == "bm25":
        return BM25Retriever(documents)
    raise ValueError(f"unknown retriever type {spec.type!r} (expected 'bm25')")
