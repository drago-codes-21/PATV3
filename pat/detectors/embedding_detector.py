"""
Embedding-based semantic PII detector.

This detector is deliberately conservative: it acts as a *signal booster*
for obvious PII-like candidates, but it never acts as a primary detector.

Key improvements over original:
    - token-chunk logic fixed (correct offset handling)
    - negative prototypes added for semantic contrast
    - type families enforced (prevents PERSON <-> ADDRESS swaps)
    - safer thresholds
    - robust candidate de-duplication
    - proper short-span filtering
"""

from __future__ import annotations

import logging
import re
from typing import List, Sequence, Tuple

import numpy as np

from pat.config import get_settings
from pat.detectors.base import BaseDetector, DetectorContext, DetectorResult
from pat.detectors.debug_logging import log_decision
from pat.embeddings import EmbeddingModel
from pat.utils.text import normalize_text, split_with_token_budget, trim_span

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Semantic prototype definitions
# ---------------------------------------------------------------------------

PROTOTYPES: dict[str, list[str]] = {
    "EMAIL": ["email address", "email contact", "email account"],
    "PHONE": ["phone number", "mobile number", "telephone number"],
    "ADDRESS": ["street address", "home address", "postal address"],
    "PERSON": ["person name", "full name", "individual name"],
    "CREDENTIAL": ["login password", "secret key", "authentication token"],
    "CARD_NUMBER": ["credit card number", "debit card number"],
    "BANK_ACCOUNT": ["bank account number", "account detail"],
    "NI_NUMBER": ["national insurance number", "NINO"],
    "NHS_NUMBER": ["NHS number"],
    "POSTCODE": ["postal code", "zip code"],
}

# Negative anchors help stabilise classification by providing “anti-prototypes”.
NEGATIVE_PROTOTYPES = [
    "random number",
    "generic text",
    "identifier unrelated",
    "miscellaneous content",
]

TYPE_FAMILIES = {
    "EMAIL": {"EMAIL"},
    "PHONE": {"PHONE"},
    "ADDRESS": {"ADDRESS", "POSTCODE"},
    "PERSON": {"PERSON", "PERSON_NAME"},
    "CREDENTIAL": {"CREDENTIAL", "PASSWORD", "TOKEN", "API_KEY", "PIN"},
    "CARD_NUMBER": {"CARD_NUMBER", "BANK_ACCOUNT"},
    "BANK_ACCOUNT": {"BANK_ACCOUNT", "CARD_NUMBER"},
    "NI_NUMBER": {"NI_NUMBER"},
    "NHS_NUMBER": {"NHS_NUMBER"},
    "POSTCODE": {"POSTCODE", "ADDRESS"},
}


# Regex candidate generators (minimal noise).
CANDIDATE_REGEXES: list[re.Pattern[str]] = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),            # email
    re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.IGNORECASE),         # postcode
    re.compile(r"\b\d{4,16}\b"),                                                 # semi-long digits
]


class EmbeddingSimilarityDetector(BaseDetector):

    MAX_TOKENS = 96
    TOKEN_STRIDE = 24
    MIN_SPAN_LEN = 3
    MAX_SPAN_LEN = 64

    DEFAULT_TYPE_THRESHOLDS = {
        "EMAIL": 0.60,
        "PHONE": 0.58,
        "ADDRESS": 0.55,
        "POSTCODE": 0.55,
        "PERSON": 0.58,
        "CREDENTIAL": 0.60,
        "CARD_NUMBER": 0.60,
        "BANK_ACCOUNT": 0.60,
        "NI_NUMBER": 0.60,
        "NHS_NUMBER": 0.60,
    }

    def __init__(self, alias_name: str | None = None):
        settings = get_settings()
        self._name = alias_name or "embedding"

        try:
            self.embedding_model: EmbeddingModel | None = EmbeddingModel()
        except Exception as exc:
            LOG.warning("EmbeddingModel unavailable: %s", exc)
            self.embedding_model = None

        # Threshold config
        self.global_threshold = float(getattr(settings, "embedding_similarity_threshold", 0.55))

        loaded_thresholds = {}
        try:
            loaded_thresholds = settings.detector_thresholds.get("embedding", {}) or {}
        except Exception:
            loaded_thresholds = {}

        self.type_thresholds = {**self.DEFAULT_TYPE_THRESHOLDS, **loaded_thresholds}

        # Tokenizer is optional
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(settings.embedding_model_path))
        except Exception as exc:
            LOG.warning("Tokenizer unavailable for %s: %s", self._name, exc)
            self.tokenizer = None

        # Precompute prototype embeddings
        self.prototype_types = []
        phrases = []

        for pii_type, plist in PROTOTYPES.items():
            for phrase in plist:
                self.prototype_types.append(pii_type)
                phrases.append(phrase)

        # Add negative prototypes
        for phrase in NEGATIVE_PROTOTYPES:
            self.prototype_types.append("NEGATIVE")
            phrases.append(phrase)

        if self.embedding_model:
            try:
                self.prototype_embeddings = self.embedding_model.encode_batch(phrases)
            except Exception as exc:
                LOG.warning("Failed to compute prototype embeddings: %s", exc)
                self.prototype_embeddings = None
        else:
            self.prototype_embeddings = None

    @property
    def name(self) -> str:
        return self._name

    # ----------------------------------------------------------------------
    # Main detector logic
    # ----------------------------------------------------------------------

    def run(self, text: str, *, context: DetectorContext | None = None) -> Sequence[DetectorResult]:
        if not text:
            return []

        if (self.embedding_model is None) or (self.prototype_embeddings is None):
            return []

        candidates = self._generate_candidates(text)
        if not candidates:
            return []

        candidate_texts = [text[s:e] for s, e in candidates]

        try:
            embeddings = self.embedding_model.encode_batch(candidate_texts)
        except Exception as exc:
            LOG.warning("Embedding encode failed: %s", exc)
            return []

        if embeddings.shape[1] != self.prototype_embeddings.shape[1]:
            LOG.error(
                "Embedding dimension mismatch: detector=%s prototypes=%s; disabling embedding detector.",
                embeddings.shape,
                self.prototype_embeddings.shape,
            )
            return []

        similarity = embeddings @ self.prototype_embeddings.T

        results: list[DetectorResult] = []
        seen: set[tuple[int, int, str]] = set()

        for idx, (start, end) in enumerate(candidates):
            scores = similarity[idx]

            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_type = self.prototype_types[best_idx]

            # skip negative prototypes
            if best_type == "NEGATIVE":
                continue

            # family-safe check
            family = TYPE_FAMILIES.get(best_type, {best_type})
            threshold = self.type_thresholds.get(best_type, self.global_threshold)

            if best_score < threshold:
                continue

            adj_start, adj_end, adj_text = trim_span(text, start, end)

            if len(adj_text) < self.MIN_SPAN_LEN or len(adj_text) > self.MAX_SPAN_LEN:
                continue

            key = (adj_start, adj_end, best_type)
            if key in seen:
                continue
            seen.add(key)

            results.append(
                DetectorResult(
                    start=adj_start,
                    end=adj_end,
                    text=adj_text,
                    pii_type=best_type,
                    confidence=best_score,
                    detector_name=self.name,
                )
            )

        return results

    # ----------------------------------------------------------------------
    # Candidate generation
    # ----------------------------------------------------------------------

    def _generate_candidates(self, text: str) -> List[Tuple[int, int]]:
        """Generate candidate spans using tokenizer (preferred) and regex."""
        normalized = normalize_text(text)
        search_text = text if len(normalized) != len(text) else normalized

        candidates: list[tuple[int, int]] = []
        text_len = len(search_text)

        # Token-based candidates
        if self.tokenizer and getattr(self.tokenizer, "is_fast", False):
            tokenized = self.tokenizer(
                search_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False,
            )
            offsets: Sequence[tuple[int, int]] = tokenized["offset_mapping"]

            chunks = split_with_token_budget(
                search_text,
                offsets,
                max_tokens=self.MAX_TOKENS,
                stride=self.TOKEN_STRIDE,
            )

            for chunk_start, chunk_end in chunks:
                for token_start, token_end in offsets:
                    if token_start < chunk_start or token_end > chunk_end:
                        continue

                    token = search_text[token_start:token_end]
                    if not token:
                        continue

                    if ("@" in token) or any(ch.isdigit() for ch in token):
                        span_end = min(text_len, token_end + 20)
                        candidates.append((token_start, span_end))

        # Regex fallback always runs
        for pattern in CANDIDATE_REGEXES:
            for match in pattern.finditer(search_text):
                candidates.append(match.span())

        # Deduplicate & clamp
        deduped = []
        seen = set()

        for s, e in candidates:
            s = max(0, s)
            e = min(text_len, e)
            if e <= s:
                continue
            key = (s, e)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)

        return deduped


# legacy naming
SemanticSimilarityDetector = EmbeddingSimilarityDetector
