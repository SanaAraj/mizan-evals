"""BM25 lexical retriever over an Arabic-normalized corpus."""

from __future__ import annotations

from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from mizan.corpus import Document
from mizan.retrieval.normalize import tokenize


class BM25Retriever:
    """Rank corpus documents against a query with Okapi BM25.

    Documents are indexed on their normalized ``title`` + ``text``. Ties in score
    are broken by original corpus order, so retrieval is deterministic.
    """

    def __init__(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("cannot build a retriever over an empty corpus")
        self._doc_ids = [doc.id for doc in documents]
        tokenized = [tokenize(f"{doc.title} {doc.text}") for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int) -> list[str]:
        """Return the ids of the top ``k`` documents for ``query``, best first.

        Raises:
            ValueError: if ``k`` is not a positive integer.
        """
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [self._doc_ids[i] for i in order[:k]]
