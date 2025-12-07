from pat.pipeline.core import run_pipeline
from pat.fusion import FusionEngine, FusedSpan
from pat.detectors.base import DetectorResult
from pat.detectors.runner import DetectorRunner
from pat.policy import PolicyEngine
from pat.severity.model import SeverityModel


class StubRunner(DetectorRunner):
    def __init__(self):
        # do not load real detectors
        self.detectors = []

    def run(self, text: str):
        spans = []
        acct_start = text.index("12345678")
        spans.append(DetectorResult(start=acct_start, end=acct_start + 8, text="12345678", pii_type="BANK_ACCOUNT", confidence=0.9, detector_name="regex"))
        email_start = text.index("john.doe@example.com")
        spans.append(DetectorResult(start=email_start, end=email_start + len("john.doe@example.com"), text="john.doe@example.com", pii_type="EMAIL", confidence=0.8, detector_name="regex"))
        pwd_start = text.index("hunter2")
        spans.append(DetectorResult(start=pwd_start, end=pwd_start + len("hunter2"), text="hunter2", pii_type="CREDENTIAL", confidence=0.95, detector_name="domain_heuristic"))
        return spans


class StubSeverity(SeverityModel):
    def __init__(self):
        # Mark as loaded so the pipeline executes the scoring path.
        self.model = True

    def predict(self, feature_vector, pii_type=None):
        # Keep the stub deterministic and high risk to exercise policy logic.
        return 0.9, "VERY_HIGH", {"VERY_HIGH": 0.9, "LOW": 0.1}


def test_e2e_pipeline_masks_high_risk_spans():
    text = "Hi John, my bank account number is 12345678 and my email is john.doe@example.com. This is my password: hunter2."
    result = run_pipeline(
        text,
        detector_runner=StubRunner(),
        fusion_engine=FusionEngine(),
        severity_model=StubSeverity(),
        policy_engine=PolicyEngine(),
    )
    sanitized = result.sanitized_text
    assert "12345678" not in sanitized
    assert "hunter2" not in sanitized
    assert "<CONTACT>" in sanitized or "<EMAIL_ADDRESS>" in sanitized
    # Ensure VERY_HIGH masked
    for span in result.spans:
        if getattr(span, "severity_label", "") == "VERY_HIGH":
            assert getattr(span, "policy_decision", {}).get("decision") == "MASK"
