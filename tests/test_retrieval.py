"""Tests for retrieval scoring metrics."""

from __future__ import annotations

import pytest

from mizan.scoring import mean_recall_at_k, mrr, recall_at_k, reciprocal_rank


class TestRecallAtK:
    def test_all_relevant_in_top_k(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_partial_recall(self) -> None:
        # One of two relevant docs is inside the top 2.
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == 0.5

    def test_denominator_is_total_relevant_not_k(self) -> None:
        # Three relevant docs but k=1 caps achievable recall at 1/3.
        assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=1) == pytest.approx(1 / 3)

    def test_duplicates_do_not_inflate_recall(self) -> None:
        # "a" repeated must count once; the second slot is a duplicate, not "b".
        assert recall_at_k(["a", "a"], {"a", "b"}, k=2) == 0.5

    def test_k_larger_than_result_list(self) -> None:
        assert recall_at_k(["a"], {"a"}, k=10) == 1.0

    def test_empty_retrieved_is_zero(self) -> None:
        assert recall_at_k([], {"a"}, k=5) == 0.0

    def test_zero_k_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            recall_at_k(["a"], {"a"}, k=0)

    def test_empty_relevant_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            recall_at_k(["a"], set(), k=3)


class TestReciprocalRank:
    def test_first_position(self) -> None:
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_third_position(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_uses_first_relevant_only(self) -> None:
        assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5

    def test_duplicates_are_collapsed_before_ranking(self) -> None:
        # Leading duplicate "x" collapses, so "a" is at rank 2, not 3.
        assert reciprocal_rank(["x", "x", "a"], {"a"}) == 0.5

    def test_no_relevant_retrieved_is_zero(self) -> None:
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_empty_relevant_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            reciprocal_rank(["a"], set())


class TestAggregates:
    def test_mean_recall_at_k(self) -> None:
        cases = [
            (["a", "b"], {"a", "b"}),  # 1.0
            (["x", "y"], {"a"}),  # 0.0
        ]
        assert mean_recall_at_k(cases, k=2) == 0.5

    def test_mrr(self) -> None:
        cases = [
            (["a"], {"a"}),  # 1.0
            (["x", "a"], {"a"}),  # 0.5
        ]
        assert mrr(cases) == 0.75

    def test_mean_recall_over_no_cases_raises(self) -> None:
        with pytest.raises(ValueError, match="zero cases"):
            mean_recall_at_k([], k=3)

    def test_mrr_over_no_cases_raises(self) -> None:
        with pytest.raises(ValueError, match="zero cases"):
            mrr([])
