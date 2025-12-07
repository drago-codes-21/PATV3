import numpy as np

from pat.severity.infer import run_severity_inference
from pat.detectors.base import DetectorResult
from pat.fusion import FusionEngine
from pat.severity.model import SeverityModel


class DummyRunner:
    def run(self, text: str):
        idx = text.index("user@example.com")
        return [
            DetectorResult(
                start=idx,
                end=idx + len("user@example.com"),
                text="user@example.com",
                pii_type="EMAIL",
                confidence=0.9,
                detector_name="regex",
            )
        ]


class DummyEmbeddingModel:
    def encode(self, text: str):
        return np.zeros(4, dtype=float)


class DummySeverityModel(SeverityModel):
    def __init__(self):
        pass

    def predict(self, feature_vector, pii_type=None):
        return 0.7, "HIGH", {"HIGH": 0.7, "MEDIUM": 0.2, "LOW": 0.1}


def test_severity_inference_basic():
    text = "Reach me at user@example.com today."
    result = run_severity_inference(
        text,
        runner=DummyRunner(),
        fusion=FusionEngine(),
        severity_model=DummySeverityModel(),
        embedding_model=DummyEmbeddingModel(),
        include_features=True,
    )
    assert result["spans"], "Should find at least one span"
    span = result["spans"][0]
    assert span["pii_type"] == "EMAIL"
    assert span["severity_label"] == "HIGH"
    assert "severity_score" in span
