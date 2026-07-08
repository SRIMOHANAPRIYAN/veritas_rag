"""Tests for span-overlap evaluation functions."""

import pytest

from evaluation.metrics import compute_span_overlap, is_span_relevant


class TestComputeSpanOverlap:
    """Tests for compute_span_overlap."""

    def test_no_overlap_before(self) -> None:
        """Retrieved span is entirely before gold span."""
        assert compute_span_overlap(100, 200, 0, 50) == 0.0

    def test_no_overlap_after(self) -> None:
        """Retrieved span is entirely after gold span."""
        assert compute_span_overlap(100, 200, 300, 400) == 0.0

    def test_exact_match(self) -> None:
        """Retrieved span exactly matches gold span."""
        assert compute_span_overlap(100, 200, 100, 200) == 1.0

    def test_partial_overlap_left(self) -> None:
        """Retrieved span overlaps the left part of gold span."""
        # Gold: [100, 200), Ret: [50, 150)
        # Overlap: [100, 150) = 50 chars, gold_length = 100
        assert compute_span_overlap(100, 200, 50, 150) == pytest.approx(0.5)

    def test_partial_overlap_right(self) -> None:
        """Retrieved span overlaps the right part of gold span."""
        # Gold: [100, 200), Ret: [150, 250)
        # Overlap: [150, 200) = 50 chars, gold_length = 100
        assert compute_span_overlap(100, 200, 150, 250) == pytest.approx(0.5)

    def test_containment_retrieved_contains_gold(self) -> None:
        """Retrieved span fully contains gold span."""
        # Gold: [100, 200), Ret: [50, 300)
        # Overlap: [100, 200) = 100 chars, gold_length = 100
        assert compute_span_overlap(100, 200, 50, 300) == 1.0

    def test_containment_gold_contains_retrieved(self) -> None:
        """Gold span fully contains retrieved span."""
        # Gold: [100, 300), Ret: [150, 200)
        # Overlap: [150, 200) = 50 chars, gold_length = 200
        assert compute_span_overlap(100, 300, 150, 200) == pytest.approx(0.25)

    def test_zero_length_gold_span(self) -> None:
        """Gold span has zero length — should return 0.0."""
        assert compute_span_overlap(100, 100, 50, 150) == 0.0

    def test_adjacent_no_overlap(self) -> None:
        """Spans are adjacent but don't overlap."""
        # Gold: [100, 200), Ret: [200, 300)
        assert compute_span_overlap(100, 200, 200, 300) == 0.0

    def test_single_char_overlap(self) -> None:
        """Spans overlap by exactly one character."""
        # Gold: [100, 200), Ret: [199, 300)
        # Overlap: [199, 200) = 1 char, gold_length = 100
        assert compute_span_overlap(100, 200, 199, 300) == pytest.approx(0.01)

    def test_small_gold_large_retrieved(self) -> None:
        """Small gold span, large retrieved span fully containing it."""
        # Gold: [100, 110), Ret: [0, 1000)
        # Overlap: 10 chars, gold_length = 10
        assert compute_span_overlap(100, 110, 0, 1000) == 1.0


class TestIsSpanRelevant:
    """Tests for is_span_relevant."""

    def test_different_doc_id(self) -> None:
        """Different doc_id should never be relevant."""
        assert not is_span_relevant("doc_a", 100, 200, "doc_b", 100, 200, 0.5)

    def test_same_doc_sufficient_overlap(self) -> None:
        """Same doc, overlap >= threshold → relevant."""
        # Gold: [100, 200), Ret: [50, 200) → overlap = 100/100 = 1.0
        assert is_span_relevant("doc_a", 100, 200, "doc_a", 50, 200, 0.5)

    def test_same_doc_insufficient_overlap(self) -> None:
        """Same doc, overlap < threshold → not relevant."""
        # Gold: [100, 200), Ret: [190, 300) → overlap = 10/100 = 0.1
        assert not is_span_relevant("doc_a", 100, 200, "doc_a", 190, 300, 0.5)

    def test_exact_threshold(self) -> None:
        """Overlap exactly equals threshold → relevant."""
        # Gold: [100, 200), Ret: [150, 250) → overlap = 50/100 = 0.5
        assert is_span_relevant("doc_a", 100, 200, "doc_a", 150, 250, 0.5)

    def test_just_below_threshold(self) -> None:
        """Overlap just below threshold → not relevant."""
        # Gold: [100, 200), Ret: [151, 251) → overlap = 49/100 = 0.49
        assert not is_span_relevant("doc_a", 100, 200, "doc_a", 151, 251, 0.5)

    def test_zero_threshold(self) -> None:
        """With threshold=0, any same-doc overlap is relevant."""
        # Gold: [100, 200), Ret: [199, 300) → overlap = 1/100 = 0.01
        assert is_span_relevant("doc_a", 100, 200, "doc_a", 199, 300, 0.0)

    def test_threshold_one_requires_full_coverage(self) -> None:
        """With threshold=1.0, only full gold coverage is relevant."""
        # Gold: [100, 200), Ret: [100, 200) → overlap = 1.0
        assert is_span_relevant("doc_a", 100, 200, "doc_a", 100, 200, 1.0)
        # Gold: [100, 200), Ret: [100, 199) → overlap = 0.99
        assert not is_span_relevant("doc_a", 100, 200, "doc_a", 100, 199, 1.0)
