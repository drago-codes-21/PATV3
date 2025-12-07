from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
import pytest
from pat.detectors.base import DetectorResult
from pat.fusion import FusedSpan
from pat.severity import (
    FEATURE_NAMES,
    SeverityModel,
    compute_neighbor_stats,
    compute_span_sentence_context,
    compute_token_position_ratio,
    extract_span_features,
    span_features_to_vector,
)
from pat.utils.text import compute_sentence_boundaries



def test_extract_span_features_includes_source_and_stats():
    span = FusedSpan(
        start=0,
        end=8,
        text="12345678",
        pii_type="BANK_ACCOUNT",
        max_confidence=0.9,
        sources=["regex"],
    )
    text = "Account 12345678 is confidential."
    features = extract_span_features(span, text, embedding=np.ones(4))
    assert features["source_regex"] == 1.0
    assert features["digit_count"] == 8.0
    vector = span_features_to_vector(features)
    assert vector.shape[0] == len(FEATURE_NAMES)
    assert features["num_detectors"] >= 1


def test_severity_model_predicts_score(tmp_path: Path):
    """Verify the SeverityModel loads a model and returns a float score."""
    # Create a dummy model that always predicts class 1 (HIGH/VERY_HIGH)
    dummy_sklearn_model = LogisticRegression()
    dummy_sklearn_model.fit(np.array([[0.0] * len(FEATURE_NAMES)]), [1])
    model_path = tmp_path / "dummy_model.joblib"
    joblib.dump(dummy_sklearn_model, model_path)

    model = SeverityModel(model_path=model_path)
    # The predict method should return the probability of the positive class (class 1)
    score, _, _ = model.predict([0.0] * len(FEATURE_NAMES))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_training_and_inference_feature_consistency(tmp_path: Path):
    """
    Tests that training and inference use the same feature path and that the
    model can be trained and used for prediction.
    """
    # 1. Create synthetic data and a FusedSpan
    text = "Contact me at user@example.com for details."
    span = FusedSpan(
        start=15,
        end=31,
        text="user@example.com",
        pii_type="EMAIL",
        confidence=0.9,
        sources=[],
    )

    # 2. Build feature vector using the shared functions
    sentences = compute_sentence_boundaries(text)
    sent_idx, sent_ratio = compute_span_sentence_context(span, sentences)
    tok_ratio = compute_token_position_ratio(text, span)

    embedding = np.random.rand(768)

    features = extract_span_features(
        span,
        text,
        sentence_index=sent_idx,
        sentence_position_ratio=sent_ratio,
        token_position_ratio=tok_ratio,
        embedding=embedding,
    )
    feature_vector = span_features_to_vector(features)
    assert feature_vector.shape[0] == len(FEATURE_NAMES)
    assert features["has_at"] == 1.0

    # 3. Train a dummy model on a minimal dataset
    X_train = np.vstack([feature_vector, np.zeros_like(feature_vector)])
    y_train = [1, 0]  # Use numeric labels for sklearn
    model = LogisticRegression()
    model.fit(X_train, y_train)
    model.classes_ = np.array([0, 1]) # Mock the classes attribute

    # 4. Save model and load it via SeverityModel for inference
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)
    severity_model = SeverityModel(model_path=model_path)

    # 5. Run inference and check the raw score
    predicted_score, _, _ = severity_model.predict(feature_vector.tolist())

    # 6. Assert that the model predicts a high probability for the class it was trained on
    assert predicted_score > 0.5


def test_keyword_and_neighbor_features():
    text = "Hi John, my bank account number is 12345678 and my password is hunter2."
    spans = [
        FusedSpan(start=text.index("John"), end=text.index("John") + 4, text="John", pii_type="PERSON", confidence=0.6),
        FusedSpan(start=text.index("12345678"), end=text.index("12345678") + 8, text="12345678", pii_type="BANK_ACCOUNT", confidence=0.9),
        FusedSpan(start=text.index("hunter2"), end=text.index("hunter2") + 7, text="hunter2", pii_type="CREDENTIAL", confidence=0.95),
    ]
    neighbor_stats = compute_neighbor_stats(spans, window_chars=32)
    sentences = compute_sentence_boundaries(text)
    features = extract_span_features(
        spans[1],
        text,
        sentence_index=compute_span_sentence_context(spans[1], sentences)[0],
        neighbor_span_count=neighbor_stats[1][0],
        neighbor_high_risk_count=neighbor_stats[1][1],
    )
    assert features["neighbor_span_count"] >= 1
    assert features["ctx_has_bank"] == 1.0
