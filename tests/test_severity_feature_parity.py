import numpy as np

from pat.fusion import FusedSpan
from pat.severity import features as severity_features
from pat.severity import train as severity_train


class ZeroEmbeddings:
    def encode_batch(self, texts):
        return np.zeros((len(texts), 4), dtype=float)

    def encode(self, text):
        return np.zeros(4, dtype=float)


def test_training_and_inference_features_match(monkeypatch):
    """
    Ensure the dataset builder and inference extraction produce identical vectors
    for the same spans, including neighbor stats.
    """
    text = "Email user@example.com and phone +44 7700 900123."
    spans = [
        FusedSpan(start=text.index("user"), end=text.index("user") + len("user@example.com"), text="user@example.com", pii_type="EMAIL", max_confidence=0.9, sources=["regex"]),
        FusedSpan(start=text.index("+44"), end=text.index("+44") + len("+44 7700 900123"), text="+44 7700 900123", pii_type="PHONE", max_confidence=0.8, sources=["regex"]),
    ]

    # Inference path
    sentences = severity_features.compute_sentence_boundaries(text)
    neighbor_stats_inf = severity_features.compute_neighbor_stats(spans)
    inf_vectors = []
    for idx, span in enumerate(spans):
        sent_idx, sent_ratio = severity_features.compute_span_sentence_context(span, sentences)
        tok_ratio = severity_features.compute_token_position_ratio(text, span)
        feat = severity_features.extract_span_features(
            span,
            text,
            sentence_index=sent_idx,
            sentence_position_ratio=sent_ratio,
            token_position_ratio=tok_ratio,
            embedding=np.zeros(4),
            context_embedding=np.zeros(4),
            neighbor_span_count=neighbor_stats_inf[idx][0],
            neighbor_high_risk_count=neighbor_stats_inf[idx][1],
        )
        inf_vectors.append(severity_features.span_features_to_vector(feat))

    # Training path via dataset builder
    monkeypatch.setattr(severity_train, "EmbeddingModel", lambda *a, **k: ZeroEmbeddings())
    builder = severity_train.SpanDatasetBuilder(use_detectors=False)
    rows = []
    for span in spans:
        rows.append(
            {
                "text": text,
                "span_start": span.start,
                "span_end": span.end,
                "span_text": span.text,
                "pii_type": span.pii_type,
                "severity_label": "MEDIUM",
                "confidence": span.max_confidence,
                "sources": ["regex"],
            }
        )
    X, _ = builder.build(
        rows,
        text_column="text",
        span_text_column="span_text",
        span_start_column="span_start",
        span_end_column="span_end",
        pii_type_column="pii_type",
        label_column="severity_label",
    )

    assert X.shape[0] == len(spans)
    # Compare vectors element-wise; neighbor stats ensure non-zero differences would show up.
    np.testing.assert_array_equal(X[0], inf_vectors[0])
    np.testing.assert_array_equal(X[1], inf_vectors[1])
