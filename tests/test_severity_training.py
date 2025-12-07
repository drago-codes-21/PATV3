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
