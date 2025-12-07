import joblib
import numpy as np
import pytest

from pat.severity import FEATURE_NAMES, SeverityModel


class _ValidDummyModel:
    def __init__(self):
        self.classes_ = np.array(["LOW", "MEDIUM", "HIGH", "VERY_HIGH"])
        self.feature_names_in_ = np.array(FEATURE_NAMES)
        self.schema_version = "severity_v1"

    def predict_proba(self, X):
        probs = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=float)
        return np.repeat(probs, len(X), axis=0)


class _BadClassModel(_ValidDummyModel):
    def __init__(self):
        super().__init__()
        self.classes_ = np.array(["LOW", "HIGH"])


def test_severity_model_validates_schema_and_classes(tmp_path):
    model_path = tmp_path / "ok_model.joblib"
    joblib.dump(_ValidDummyModel(), model_path)

    severity = SeverityModel(model_path=model_path)
    score, label, probs = severity.predict([0.0] * len(FEATURE_NAMES))
    assert label == "VERY_HIGH"
    assert score == pytest.approx(0.75)
    assert set(probs.keys()) == {"LOW", "MEDIUM", "HIGH", "VERY_HIGH"}


def test_severity_model_raises_on_bad_classes(tmp_path):
    model_path = tmp_path / "bad_model.joblib"
    joblib.dump(_BadClassModel(), model_path)

    with pytest.raises(ValueError):
        SeverityModel(model_path=model_path)
