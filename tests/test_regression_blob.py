import re
from pathlib import Path

import numpy as np

from pat.detectors.base import DetectorResult
from pat.detectors.regex_detector import RegexDetector
from pat.fusion import FusionEngine
from pat.pipeline import RedactionPipeline
from pat.policy import PolicyEngine
from pat.policy.engine import PolicyDecision
from pat.pipeline.pipeline import _apply_redactions
from pat.fusion.span import FusedSpan


def test_regex_covers_high_risk_ids_and_credentials():
    text = """
    NI: QQ 98 76 54 A
    NHS: 987 654 3210
    DOB: 1991-02-14
    postgres://pii_admin:Sup3rS3cretP@localhost:5432/pii_prod
    ADMIN_PASSWORD="Welcome123!"
    PII_API_KEY = "sk_test_9a8b7c6d5e4f3g2h1x0y"
    Health: chronic asthma and mild depression noted.
    """
    detector = RegexDetector()
    results = detector.run(text)
    types = {r.pii_type for r in results}
    assert "NI_NUMBER" in types
    assert "NHS_NUMBER" in types
    assert "DATE" in types
    assert "CREDENTIAL" in types
    assert "API_KEY" in types
    assert any("asthma" in r.text.lower() for r in results if r.pii_type == "MEDICAL_INFO")
    assert any(r.text.startswith("postgres://") for r in results if r.pii_type == "CREDENTIAL")


class _StaticSeverity:
    def __init__(self, score: float = 0.8, label: str = "HIGH") -> None:
        self.model = True
        self.score = score
        self.label = label

    def predict(self, feature_vector, pii_type=None):
        return self.score, self.label, {self.label: 0.9}


class _ZeroEmbeddings:
    def encode_batch(self, texts):
        return np.zeros((len(texts), 4), dtype=float)

    def encode(self, text):
        return np.zeros(4, dtype=float)


def test_single_pass_masking_preserves_structure():
    text = (
        "Billing Address: 18 Maple Street,\n"
        "Billing Email: michael.turner.personal@gmail.com\n"
        "Emergency Contact Phone: +44 7700 812345\n"
        "Employee ID: EMP-00127\n"
    )

    spans = [
        FusedSpan(
            start=text.index("18"),
            end=text.index("18") + len("18 Maple Street"),
            text="18 Maple Street",
            pii_type="ADDRESS",
        ),
        FusedSpan(
            start=text.index("michael.turner.personal@gmail.com"),
            end=text.index("michael.turner.personal@gmail.com") + len("michael.turner.personal@gmail.com"),
            text="michael.turner.personal@gmail.com",
            pii_type="EMAIL",
        ),
        FusedSpan(
            start=text.index("+44 7700 812345"),
            end=text.index("+44 7700 812345") + len("+44 7700 812345"),
            text="+44 7700 812345",
            pii_type="PHONE",
        ),
        FusedSpan(
            start=text.index("EMP-00127"),
            end=text.index("EMP-00127") + len("EMP-00127"),
            text="EMP-00127",
            pii_type="EMPLOYMENT_ID",
        ),
    ]

    decisions = [
        PolicyDecision(span=span, action="MASK", rule_id="test", placeholder="<ADDRESS>")
        if span.pii_type == "ADDRESS"
        else PolicyDecision(
            span=span,
            action="MASK",
            rule_id="test",
            placeholder="<CONTACT>" if span.pii_type in {"EMAIL", "PHONE"} else "<EMPLOYMENT>",
        )
        for span in spans
    ]

    sanitized = _apply_redactions(text, decisions)

    assert "Billing Address: <ADDRESS>" in sanitized
    assert "Billing Email: <CONTACT>" in sanitized
    assert "Emergency Contact Phone: <CONTACT>" in sanitized
    assert "Employee ID: <EMPLOYMENT>" in sanitized
    # Ensure no corrupted fragments from overlapping replacements
    assert "<ADDRESS>g Email" not in sanitized
    assert "<PERSON>DRESS" not in sanitized


def test_end_to_end_blob_regression():
    blob_path = Path("samples/test_blob.txt")
    raw = blob_path.read_text(encoding="utf-8")

    pipeline = RedactionPipeline(
        detector_runner=None,  # default detectors
        fusion_engine=FusionEngine(),
        severity_model=_StaticSeverity(score=0.82, label="HIGH"),
        policy_engine=PolicyEngine(),
        embedding_model=_ZeroEmbeddings(),
    )

    result = pipeline.run(raw)
    sanitized = result["sanitized_text"]

    # Critical PII must be masked
    assert "QQ 12 34 56 C" not in sanitized
    assert "4485 9932 1184 2231" not in sanitized
    assert "postgres://pii_admin" not in sanitized
    assert "sk_test_9a8b7c6d5e4f3g2h1x0y" not in sanitized
    assert "Password@123" not in sanitized
    assert "987 654 3210" not in sanitized
    assert "1991-02-14" not in sanitized

    # Structure and non-PII should remain readable
    assert "ORD-20240910-000142" in sanitized
    assert "[EMAIL THREAD – CUSTOMER SUPPORT]" in sanitized
    assert "MIGRATION TO PAT" in sanitized

    # No corrupted placeholder fragments
    assert "<PERSON>DRESS" not in sanitized
    assert "<ADDRESS>g Email" not in sanitized
    assert "><EMPLOYMENT>>" not in sanitized
