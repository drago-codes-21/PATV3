from pat.detectors.base import DetectorResult
from pat.fusion import FusionEngine


def test_fusion_merges_overlapping_spans():
    engine = FusionEngine()
    detector_results = [
        DetectorResult(
            start=0,
            end=5,
            text="Alice",
            pii_type="PERSON",
            confidence=0.7,
            detector_name="ner",
        ),
        DetectorResult(
            start=2,
            end=7,
            text="Alice ",
            pii_type="PERSON",
            confidence=0.9,
            detector_name="regex",
        ),
    ]
    fused = engine.fuse(detector_results)
    assert len(fused) == 1
    assert fused[0].start == 0
    assert fused[0].end == 7
    assert fused[0].pii_type == "PERSON"
    assert len(fused[0].sources) == 2


def test_fusion_groups_adjacent_person_tokens():
    engine = FusionEngine()
    text = "John Smith visited."
    results = [
        DetectorResult(start=0, end=4, text="John", pii_type="PERSON", confidence=0.6, detector_name="ner"),
        DetectorResult(start=5, end=10, text="Smith", pii_type="PERSON", confidence=0.65, detector_name="ner"),
    ]
    fused = engine.fuse(results, text=text)
    assert len(fused) == 1
    assert fused[0].start == 0 and fused[0].end >= 10
    assert fused[0].pii_type.startswith("PERSON")


def test_fusion_prefers_specific_over_generic_number():
    engine = FusionEngine()
    num_start, num_end = 25, 33
    results = [
        DetectorResult(start=num_start, end=num_end, text="12345678", pii_type="GENERIC_NUMBER", confidence=0.4, detector_name="regex"),
        DetectorResult(start=num_start, end=num_end, text="12345678", pii_type="BANK_ACCOUNT", confidence=0.8, detector_name="domain_heuristic"),
    ]
    fused = engine.fuse(results)
    assert len(fused) == 1
    assert fused[0].pii_type == "BANK_ACCOUNT"
    assert fused[0].confidence >= 0.8


def test_fusion_regression_sample_sentence():
    engine = FusionEngine()
    text = (
        "Hi John, my bank account number is 12345678 and my email is john.doe@example.com. "
        "Please don’t share it."
    )
    acct_idx = text.index("12345678")
    email_idx = text.index("john.doe@example.com")
    results = [
        DetectorResult(start=text.index("John"), end=text.index("John") + 4, text="John", pii_type="PERSON", confidence=0.6, detector_name="ml_ner"),
        DetectorResult(start=acct_idx, end=acct_idx + 8, text="12345678", pii_type="BANK_ACCOUNT", confidence=0.9, detector_name="regex"),
        DetectorResult(start=email_idx, end=email_idx + len("john.doe@example.com"), text="john.doe@example.com", pii_type="EMAIL", confidence=0.95, detector_name="regex"),
    ]
    fused = engine.fuse(results, text=text)
    assert len(fused) >= 2
    types = {span.pii_type for span in fused}
    assert "BANK_ACCOUNT" in types
    assert "EMAIL" in types
    # Ensure spans are not merged into a single oversized span
    assert max(span.end for span in fused) - min(span.start for span in fused) < len(text)
