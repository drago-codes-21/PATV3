"""Unit tests for the FusionEngine."""

import pytest

from pat.detectors.base import DetectorResult
from pat.fusion import FusionEngine
from pat.fusion.span import FusedSpan


def test_fuse_non_overlapping_spans():
    """
    Verify that non-overlapping spans remain separate. This was the primary
    bug case that caused incorrect merging.
    """
    engine = FusionEngine()
    results = [
        DetectorResult(
            start=25, end=33, text="12345678", pii_type="BANK_ACCOUNT", confidence=0.9, detector_name="regex"
        ),
        DetectorResult(
            start=52, end=68, text="test@example.com", pii_type="EMAIL", confidence=0.9, detector_name="regex"
        ),
    ]
    fused = engine.fuse(results)

    assert len(fused) == 2, "Should produce two separate spans"
    assert fused[0].start == 25
    assert fused[0].end == 33
    assert fused[0].pii_type == "BANK_ACCOUNT"
    assert fused[1].start == 52
    assert fused[1].end == 68
    assert fused[1].pii_type == "EMAIL"


def test_fuse_overlapping_spans_with_different_types():
    """Verify that overlapping spans are correctly merged."""
    engine = FusionEngine()
    results = [
        DetectorResult(start=10, end=20, text="...", pii_type="A", confidence=0.8, detector_name="d1"),
        DetectorResult(start=15, end=25, text="...", pii_type="B", confidence=0.9, detector_name="d2"),
    ]
    fused = engine.fuse(results)

    assert len(fused) == 1, "Should merge into a single span"
    span = fused[0]
    assert span.start == 10
    assert span.end == 25
    assert span.pii_type == "B", "Dominant type should win after conflict resolution"
    assert span.all_types == {"A", "B"}
    assert span.confidence >= 0.9, "Confidence should reflect top weighted source"
    assert len(span.sources) == 2


def test_fuse_adjacent_spans_do_not_merge():
    """Verify that spans that touch but do not overlap remain separate."""
    engine = FusionEngine()
    results = [
        DetectorResult(start=10, end=20, text="first", pii_type="A", confidence=0.9, detector_name="d1"),
        DetectorResult(start=20, end=30, text="second", pii_type="B", confidence=0.9, detector_name="d2"),
    ]
    fused = engine.fuse(results)

    assert len(fused) == 2, "Adjacent spans should not merge"
    assert fused[0].end == 20
    assert fused[1].start == 20


def test_fuse_contained_span():
    """Verify a smaller span is correctly merged into a larger one."""
    engine = FusionEngine()
    results = [
        DetectorResult(start=10, end=30, text="...", pii_type="ADDRESS", confidence=0.7, detector_name="d1"),
        DetectorResult(start=15, end=20, text="...", pii_type="POSTCODE", confidence=0.9, detector_name="d2"),
    ]
    fused = engine.fuse(results)

    assert len(fused) == 1
    span = fused[0]
    assert span.start == 10
    assert span.end == 30
    assert span.pii_type in {"ADDRESS", "POSTCODE"}
    assert span.all_types == {"ADDRESS", "POSTCODE"}
    assert span.confidence >= 0.9


def test_fuse_empty_and_single_result():
    """Test edge cases of empty or single-item lists."""
    engine = FusionEngine()
    assert engine.fuse([]) == []

    single_result = [DetectorResult(start=0, end=5, text="...", pii_type="A", confidence=0.5, detector_name="d1")]
    fused = engine.fuse(single_result)
    assert len(fused) == 1
    assert fused[0].start == 0
    assert fused[0].end == 5


def test_complex_case_with_multiple_merges_and_gaps():
    """Test a more complex scenario with out-of-order results."""
    engine = FusionEngine()
    results = [
        DetectorResult(start=100, end=110, text="...", pii_type="C", confidence=0.8, detector_name="d3"),
        DetectorResult(start=10, end=20, text="...", pii_type="A", confidence=0.8, detector_name="d1"),
        DetectorResult(start=18, end=25, text="...", pii_type="B", confidence=0.9, detector_name="d2"),
    ]
    fused = engine.fuse(results)

    assert len(fused) == 2
    assert fused[0].start == 10 and fused[0].end == 25
    assert fused[1].start == 100 and fused[1].end == 110 and fused[1].pii_type == "C"
