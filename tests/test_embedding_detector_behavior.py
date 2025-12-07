import numpy as np
import pytest

from pat.detectors.base import DetectorContext, DetectorResult
from pat.detectors.embedding_detector import EmbeddingSimilarityDetector


class DummyEmbeddingModel:
    def encode_batch(self, texts):
        return np.zeros((len(texts), 3), dtype=float)


class DummyTokenizer:
    is_fast = False
    model_max_length = 32

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def test_embedding_detector_requires_similarity_and_does_not_seed(monkeypatch):
    import pat.detectors.embedding_detector as emb_mod

    monkeypatch.setattr(emb_mod, "EmbeddingModel", lambda *a, **k: DummyEmbeddingModel())
    monkeypatch.setattr(emb_mod, "AutoTokenizer", DummyTokenizer, raising=False)

    detector = EmbeddingSimilarityDetector(alias_name="embedding")
    # Normalise prototype embeddings to dummy dimension
    detector.prototype_embeddings = np.zeros(
        (len(detector.prototype_phrases), 3), dtype=float
    )
    detector.threshold = 0.5

    seed = DetectorResult(
        start=10,
        end=24,
        text="user@example.com",
        pii_type="EMAIL",
        confidence=0.9,
        detector_name="regex",
    )
    ctx = DetectorContext(prior_results=[seed])
    # All similarities are zero; detector must not emit anything, nor re-emit seeds.
    results = detector.run("Contact: user@example.com", context=ctx)
    assert results == []
