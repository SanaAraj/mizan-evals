"""Retrieval systems under test and their construction."""

from typing import Protocol, runtime_checkable

from mizan.retrieval.bm25 import BM25Retriever
from mizan.retrieval.factory import build_retriever
from mizan.retrieval.normalize import normalize_arabic, tokenize


@runtime_checkable
class Retriever(Protocol):
    """Anything that returns ranked document ids for a query."""

    def retrieve(self, query: str, k: int) -> list[str]: ...


__all__ = [
    "BM25Retriever",
    "Retriever",
    "build_retriever",
    "normalize_arabic",
    "tokenize",
]
