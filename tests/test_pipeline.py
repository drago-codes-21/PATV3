import numpy as np
from pathlib import Path

from pat.detectors.base import DetectorResult
from pat.detectors.regex_detector import RegexDetector
from pat.fusion import FusionEngine
from pat.pipeline import RedactionPipeline
from pat.policy import PolicyEngine


class DummyRunner:
    def run(self, text: str):
        idx = text.index("12345678")
        return [
            DetectorResult(
                start=idx,
                end=idx + 8,
                text="12345678",
                pii_type="BANK_ACCOUNT",
                confidence=0.95,
                detector_name="regex",
            ),
            DetectorResult(
                start=idx,
                end=idx + 8,
                text="12345678",
                pii_type="BANK_ACCOUNT",
                confidence=0.7,
                detector_name="domain",
            ),
        ]


class DummyEmbeddingModel:
    def encode(self, text: str):
        return np.ones(4, dtype=float)

    def encode_batch(self, texts):
        return np.ones((len(texts), 4), dtype=float)


class DummySeverityModel:
    def predict(self, feature_vector, pii_type=None):
        # This dummy model always returns a high score.
        return 0.9, "VERY_HIGH", {"VERY_HIGH": 0.9}

    def __init__(self):
        self.model = True # Mock that the model is loaded

def test_pipeline_masks_financial_data():
    pipeline = RedactionPipeline(
        detector_runner=DummyRunner(),
        fusion_engine=FusionEngine(),
        severity_model=DummySeverityModel(),
        policy_engine=PolicyEngine(),
        embedding_model=DummyEmbeddingModel(),
    )
    text = "Account number 12345678 is confidential."
    result = pipeline.run(text, context={"channel": "EMAIL_OUTBOUND"})
    sanitized = result["sanitized_text"]
    assert sanitized != text
    assert "<FINANCIAL>" in sanitized
    assert "12345678" not in sanitized
    assert result["severity_label"] in {"HIGH", "VERY_HIGH"}
    assert result["severity_score"] >= 0.8
    assert result["pii_spans"][0].text == "12345678"


class RegexOnlyRunner:
    def __init__(self):
        self.detector = RegexDetector()

    def run(self, text: str):
        return self.detector.run(text)


class ZeroVectorEmbeddings:
    def encode_batch(self, texts):
        return np.zeros((len(texts), 4), dtype=float)

    def encode(self, text):
        return np.zeros(4, dtype=float)


class ConstantSeverityModel:
    def __init__(self, score: float = 0.75, label: str = "HIGH"):
        self.score = score
        self.label = label
        self.model = True

    def predict(self, feature_vector, pii_type=None):
        return self.score, self.label, {self.label: 0.8}


def test_sample_text_is_sanitized():
    text = Path("samples/input.txt").read_text(encoding="utf-8")
    pipeline = RedactionPipeline(
        detector_runner=RegexOnlyRunner(),
        fusion_engine=FusionEngine(),
        severity_model=ConstantSeverityModel(),
        policy_engine=PolicyEngine(),
        embedding_model=ZeroVectorEmbeddings(),
    )
    result = pipeline.run(text)
    sanitized = result["sanitized_text"]

    assert sanitized != text
    assert "<FINANCIAL>" in sanitized
    assert "<CONTACT>" in sanitized
    assert "11892347" not in sanitized
    assert "john.doe.personal@outlook.com" not in sanitized


class PersonRunner:
    def run(self, text: str):
        idx = text.index("John Doe")
        return [
            DetectorResult(
                start=idx,
                end=idx + len("John Doe"),
                text="John Doe",
                pii_type="PERSON",
                confidence=0.9,
                detector_name="ner",
            )
        ]


def test_person_names_are_masked():
    text = "Customer John Doe reported an issue."
    pipeline = RedactionPipeline(
        detector_runner=PersonRunner(),
        fusion_engine=FusionEngine(),
        severity_model=ConstantSeverityModel(score=0.6, label="HIGH"),
        policy_engine=PolicyEngine(),
        embedding_model=ZeroVectorEmbeddings(),
    )
    result = pipeline.run(text)
    sanitized = result["sanitized_text"]
    assert sanitized != text
    assert "<PERSON>" in sanitized
    assert "John Doe" not in sanitized
