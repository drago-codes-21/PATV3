import json
from pathlib import Path

import numpy as np

from pat.severity import train as severity_train


class DummyEmbeddingModel:
    def encode_batch(self, texts):
        return np.zeros((len(texts), len(severity_train.FEATURE_NAMES)), dtype=float)[:, :4]


def test_severity_training_smoke(tmp_path, monkeypatch):
    data = [
        {
            "text": "Email user@example.com here",
            "span_start": 6,
            "span_end": 22,
            "span_text": "user@example.com",
            "pii_type": "EMAIL",
            "severity_label": "MEDIUM",
        },
        {
            "text": "Account 12345678",
            "span_start": 8,
            "span_end": 16,
            "span_text": "12345678",
            "pii_type": "BANK_ACCOUNT",
            "severity_label": "HIGH",
        },
        {
            "text": "Phone +44 7700 900123",
            "span_start": 6,
            "span_end": 19,
            "span_text": "+44 7700 900123",
            "pii_type": "PHONE",
            "severity_label": "MEDIUM",
        },
        {
            "text": "Card 4111 1111 1111 1111",
            "span_start": 5,
            "span_end": 24,
            "span_text": "4111 1111 1111 1111",
            "pii_type": "CARD_NUMBER",
            "severity_label": "HIGH",
        },
        {
            "text": "Low risk context only",
            "span_start": 0,
            "span_end": 3,
            "span_text": "Low",
            "pii_type": "GENERIC_NUMBER",
            "severity_label": "LOW",
        },
        {
            "text": "Critical secret token ABCDEF",
            "span_start": 23,
            "span_end": 29,
            "span_text": "ABCDEF",
            "pii_type": "CREDENTIAL",
            "severity_label": "VERY_HIGH",
        },
        {
            "text": "Another low item",
            "span_start": 0,
            "span_end": 7,
            "span_text": "Another",
            "pii_type": "GENERIC_NUMBER",
            "severity_label": "LOW",
        },
        {
            "text": "Another critical code ZZZ999",
            "span_start": 24,
            "span_end": 30,
            "span_text": "ZZZ999",
            "pii_type": "CREDENTIAL",
            "severity_label": "VERY_HIGH",
        },
    ]
    data_path = tmp_path / "train.jsonl"
    data_path.write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
    model_out = tmp_path / "severity_model.joblib"
    report_dir = tmp_path / "reports"

    # Prevent loading heavy models
    monkeypatch.setattr(severity_train, "EmbeddingModel", lambda *a, **k: DummyEmbeddingModel())

    severity_train.train(
        [
            "--input",
            str(data_path),
            "--output",
            str(model_out),
            "--report-dir",
            str(report_dir),
            "--disable-detector-augmentation",
            "--validation-split-size",
            "0.5",
            "--cv-folds",
            "2",
        ]
    )
    assert model_out.exists()
    assert (report_dir / f"{model_out.stem}_feature_importance.json").exists()
    assert (report_dir / f"{model_out.stem}_training_stats.json").exists()
    # Model should load through the SeverityModel interface (metadata validation)
    from pat.severity.model import SeverityModel

    sev_model = SeverityModel(model_path=model_out)
    # run a trivial predict to ensure the model is usable
    sev_model.predict([0.0] * len(severity_train.FEATURE_NAMES))
