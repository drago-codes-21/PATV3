import numpy as np

from pat.detectors.base import DetectorContext, DetectorResult
from pat.detectors.domain_heuristics_detector import DomainHeuristicsDetector
from pat.detectors.embedding_detector import EmbeddingSimilarityDetector
from pat.detectors.ner_detector import NERDetector
from pat.detectors.regex_detector import RegexDetector


def test_regex_detector_identifies_sort_code_and_account_number():
    detector = RegexDetector()
    text = "Please use sort code 20-00-00 and account number 12345678 for the transfer."
    results = detector.run(text)
    pii_types = {res.pii_type for res in results}
    assert "SORT_CODE" in pii_types
    assert "BANK_ACCOUNT" in pii_types


def test_regex_detector_rejects_random_digits():
    detector = RegexDetector()
    text = "Reference 12345678 without account keyword should not match."
    results = detector.run(text)
    assert not results


def test_regex_detector_valid_gb_iban():
    detector = RegexDetector()
    text = "IBAN: GB29NWBK60161331926819 is provided."
    results = detector.run(text)
    assert any(r.pii_type in {"BANK_ACCOUNT", "IBAN"} for r in results)


def test_regex_detector_invalid_gb_iban_not_detected():
    detector = RegexDetector()
    text = "IBAN: GB29NWBK60161331926818 (bad checksum)"
    results = detector.run(text)
    assert not results


def test_domain_heuristics_flags_credentials():
    detector = DomainHeuristicsDetector()
    text = "The customer shared their password 1234 near the login page."
    results = detector.run(text)
    assert any(res.pii_type == "CREDENTIAL" for res in results)


def test_regex_detector_ignores_non_bank_entities():
    detector = RegexDetector()
    text = "Email support@example.com or jane.doe@bank.co.uk, phone +1 (415) 555-2671."
    results = detector.run(text)
    pii_types = {r.pii_type for r in results}
    assert "PHONE" in pii_types
    assert "EMAIL" in pii_types


def test_regex_detector_requires_context_for_generic_digits():
    detector = RegexDetector()
    text = "Logged event 12345678 in the system."
    results = detector.run(text)
    assert not any(r.pii_type == "GENERIC_NUMBER" for r in results)


def test_domain_heuristics_detects_bank_account_near_keyword():
    detector = DomainHeuristicsDetector()
    text = "Please note my bank account number is 12345678 for the transfer."
    results = detector.run(text)
    assert any(res.pii_type == "BANK_ACCOUNT" for res in results)


class _DummyEnt:
    def __init__(self, start: int, end: int, label: str, score: float = 0.8) -> None:
        self.start_char = start
        self.end_char = end
        self.label_ = label
        self.score = score


class _DummyDoc:
    def __init__(self, ents):
        self.ents = ents


def test_ner_detector_maps_labels_without_spacy():
    detector = NERDetector()
    detector.nlp = lambda text: _DummyDoc([_DummyEnt(0, 4, "PERSON")])  # type: ignore[assignment]
    results = detector.run("John went home.")
    assert results and results[0].pii_type == "PERSON"


class _FakeEmbeddingModel:
    def encode_batch(self, texts):
        vecs = []
        for t in texts:
            lower = t.lower()
            vecs.append(
                [
                    1.0 if "email" in lower else 0.0,
                    1.0 if "phone" in lower or "mobile" in lower else 0.0,
                    1.0 if "account" in lower else 0.0,
                ]
            )
        return np.asarray(vecs, dtype=float)

    def encode(self, text):
        return self.encode_batch([text])[0]


def test_semantic_detector_confirms_email_with_seed():
    seed = DetectorResult(
        start=17,
        end=39,
        text="john.doe@example.com",
        pii_type="EMAIL",
        confidence=0.4,
        detector_name="regex",
    )
    detector = EmbeddingSimilarityDetector()
    detector.embedding_model = _FakeEmbeddingModel()  # type: ignore[assignment]
    text = "My email address is john.doe@example.com for contact."
    results = detector.run(text, context=DetectorContext(prior_results=[seed]))
    assert any(res.pii_type == "EMAIL" for res in results)
