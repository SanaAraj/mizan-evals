"""Retrieval metrics: recall@k and mean reciprocal rank.

Conventions used throughout:

- ``retrieved`` is an ordered sequence of document ids, best-ranked first.
  Duplicate ids are collapsed to their first occurrence before scoring, so a
  system cannot inflate a metric by returning the same document twice.
- ``relevant`` is the set of gold document ids for a query and must be
  non-empty (recall is undefined otherwise).
- ``recall@k`` uses the total number of relevant documents as the denominator
  (the textbook definition): ``|relevant ∩ top-k| / |relevant|``. When there
  are more relevant documents than ``k``, the maximum achievable value is
  therefore below 1.0.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def _dedupe(items: Sequence[str]) -> list[str]:
    """Return ``items`` with duplicates removed, preserving first-seen order."""
    return list(dict.fromkeys(items))


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant documents found within the top ``k`` retrieved.

    Args:
        retrieved: ranked document ids, best first.
        relevant: gold document ids (must be non-empty).
        k: cutoff rank; must be a positive integer.

    Raises:
        ValueError: if ``k`` is not positive or ``relevant`` is empty.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("relevant set must be non-empty; recall is undefined otherwise")
    top_k = _dedupe(retrieved)[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal of the rank of the first relevant document (0.0 if none).

    Raises:
        ValueError: if ``relevant`` is empty.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("relevant set must be non-empty; reciprocal rank is undefined otherwise")
    for rank, doc_id in enumerate(_dedupe(retrieved), start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def mean_recall_at_k(cases: Iterable[tuple[Sequence[str], Iterable[str]]], k: int) -> float:
    """Mean recall@k over ``(retrieved, relevant)`` cases.

    Raises:
        ValueError: if ``cases`` is empty (or any case is invalid).
    """
    scores = [recall_at_k(retrieved, relevant, k) for retrieved, relevant in cases]
    if not scores:
        raise ValueError("cannot average over zero cases")
    return sum(scores) / len(scores)


def mrr(cases: Iterable[tuple[Sequence[str], Iterable[str]]]) -> float:
    """Mean reciprocal rank over ``(retrieved, relevant)`` cases.

    Raises:
        ValueError: if ``cases`` is empty (or any case is invalid).
    """
    scores = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in cases]
    if not scores:
        raise ValueError("cannot average over zero cases")
    return sum(scores) / len(scores)
