"""Token-level classifier backed by mpnet embeddings."""

from __future__ import annotations

import logging
import re
from typing import List, Sequence, Tuple

import numpy as np

from .base import BaseDetector, DetectorContext, DetectorResult
from .debug_logging import log_decision
from pat.config import get_settings
from pat.embeddings import EmbeddingModel
from pat.utils.text import normalize_text, split_with_token_budget, trim_span

LOG = logging.getLogger(__name__)

# Prototype phrases describing the *semantics* of each PII type.
# The detector compares candidate spans against these using cosine similarity.
TOKEN_PROTOTYPES: dict[str, list[str]] = {
    "CARD_NUMBER": ["16 digit card number", "long card number", "credit card digits"],
    "BANK_ACCOUNT": ["bank account digits", "eight digit account number"],
    "SORT_CODE": ["bank sort code", "sort code digits"],
    "PHONE": ["mobile phone digits", "phone contact number"],
    "EMAIL": ["email address token", "email handle"],
    "ADDRESS": ["street and house number", "address line"],
    "POSTCODE": ["uk postcode token", "postal code"],
    "NI_NUMBER": ["national insurance number", "ni number"],
    "NHS_NUMBER": ["nhs service number", "national health service number digits"],
    "CREDENTIAL": ["one time password code", "temporary passcode", "login passcode"],
}

# Regex-based fallback candidates to ensure coverage when tokenization is
# unavailable or when we need additional hints.
CANDIDATE_REGEXES: list[re.Pattern[str]] = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # email-like
    re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.IGNORECASE),  # UK postcode
    re.compile(r"(?<!\d)(\d{2}[-\s]?\d{2}[-\s]?\d{2})(?!\d)"),  # sort code-like
    re.compile(r"\b\d{6,}\b"),  # generic long numbers (guarded later by similarity)
]


class MLTokenClassifierDetector(BaseDetector):
    """Classify token-level spans using mpnet embeddings.

    Design goals:
        - Act as a *semantic booster* for regex and heuristics, not as the
          primary detector.
        - Be conservative on span length to avoid over-masking.
        - Degrade gracefully when embeddings or tokenizer are unavailable.
        - Make thresholding and near-threshold behavior inspectable via logging.
    """

    MAX_TOKENS = 128
    TOKEN_STRIDE = 24
    DEFAULT_THRESHOLD = 0.48
    TYPE_THRESHOLDS: dict[str, float] = {
        "CARD_NUMBER": 0.55,
        "BANK_ACCOUNT": 0.50,
        "SORT_CODE": 0.50,
        "CREDENTIAL": 0.48,
        "PHONE": 0.50,
        "EMAIL": 0.52,
        "POSTCODE": 0.48,
        "NI_NUMBER": 0.50,
        "NHS_NUMBER": 0.50,
        "ADDRESS": 0.45,
    }

    @property
    def name(self) -> str:
        return "ml_token"

    def __init__(self) -> None:
        settings = get_settings()

        # Embedding backend. If this fails, we disable the detector gracefully.
        try:
            self.embedding_model: EmbeddingModel | None = EmbeddingModel()
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("EmbeddingModel unavailable for ml_token detector: %s", exc)
            self.embedding_model = None

        # Load per-type thresholds from config (if present).
        loaded_thresholds: dict[str, float] = {}
        try:
            if hasattr(settings, "detector_thresholds_path") and settings.detector_thresholds_path.exists():
                import json as _json

                loaded = _json.loads(settings.detector_thresholds_path.read_text(encoding="utf-8"))
                loaded_thresholds = loaded.get("ml_token", {}) or {}
        except Exception:  # pragma: no cover - config is optional
            if hasattr(settings, "detector_thresholds"):
                loaded_thresholds = settings.detector_thresholds.get("ml_token", {}) or {}

        # Merge defaults with loaded thresholds (loaded wins).
        self.thresholds: dict[str, float] = {**self.TYPE_THRESHOLDS, **loaded_thresholds}

        # Tokenizer is optional; we fall back to pure regex candidates if unavailable.
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(settings.embedding_model_path))
            max_len = getattr(self.tokenizer, "model_max_length", 512) or 512
            self.max_tokens = max(48, min(self.MAX_TOKENS, int(max_len)))
        except Exception as exc:  # pragma: no cover - optional dependency
            LOG.warning("Tokenizer unavailable for ml_token detector: %s", exc)
            self.tokenizer = None
            self.max_tokens = self.MAX_TOKENS

        # Precompute prototype embeddings once.
        self.prototype_types: list[str] = []
        prototype_phrases: list[str] = []
        for pii_type, phrases in TOKEN_PROTOTYPES.items():
            for phrase in phrases:
                self.prototype_types.append(pii_type)
                prototype_phrases.append(phrase)

        if self.embedding_model is not None and prototype_phrases:
            LOG.info("Pre-computing %d token prototype embeddings...", len(prototype_phrases))
            try:
                self.prototype_embeddings = self.embedding_model.encode_batch(prototype_phrases)
            except Exception as exc:  # pragma: no cover - defensive
                LOG.warning("Failed to compute prototype embeddings for ml_token: %s", exc)
                self.prototype_embeddings = None
        else:
            self.prototype_embeddings = None

    def run(self, text: str, *, context: DetectorContext | None = None) -> Sequence[DetectorResult]:
        """Score candidate spans and emit high-confidence PII predictions."""
        if not text:
            return []

        if self.embedding_model is None or self.prototype_embeddings is None:
            # Detector is effectively disabled, but we keep it quiet in normal runs.
            LOG.debug("ml_token detector disabled due to missing embeddings/prototypes.")
            return []

        candidates = self._generate_candidates(text)
        if not candidates:
            return []

        search_text = text  # candidates are defined against this text
        candidate_texts = [search_text[s:e] for s, e in candidates]

        try:
            embeddings = self.embedding_model.encode_batch(candidate_texts)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("Embedding encoding failed in ml_token detector: %s", exc)
            return []

        # Dimension safety: if something is misconfigured, bail out rather than
        # emitting nonsense scores.
        if embeddings.shape[1] != self.prototype_embeddings.shape[1]:
            LOG.error(
                "Embedding dimension mismatch in ml_token detector: %s vs %s; "
                "skipping detection.",
                embeddings.shape,
                self.prototype_embeddings.shape,
            )
            return []

        similarity = embeddings @ self.prototype_embeddings.T

        results: list[DetectorResult] = []
        seen: set[Tuple[int, int, str]] = set()

        for idx, (start, end) in enumerate(candidates):
            scores = similarity[idx]
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            pii_type = self.prototype_types[best_idx]

            threshold = self.thresholds.get(pii_type, self.DEFAULT_THRESHOLD)

            # Debug logging for calibration / CIP near-threshold behavior.
            if abs(best_score - threshold) <= 0.05:
                log_decision(
                    detector_name=self.name,
                    action="near_threshold",
                    text_window=search_text[start:end][:160],
                    span_before=(start, end),
                    score=best_score,
                    details={"pii_type": pii_type, "threshold": threshold},
                )

            if best_score < threshold:
                continue

            adj_start, adj_end, adj_text = trim_span(search_text, start, end)

            # Length guard: avoid emitting absurdly short or long spans.
            if len(adj_text) < 3 or len(adj_text) > 64:
                continue
            # Avoid mislabeling numeric counters/IDs as dates.
            if pii_type == "DATE" and adj_text.isdigit():
                continue

            key = (adj_start, adj_end, pii_type)
            if key in seen:
                continue
            seen.add(key)

            results.append(
                DetectorResult(
                    start=adj_start,
                    end=adj_end,
                    text=adj_text,
                    pii_type=pii_type,
                    confidence=best_score,
                    detector_name=self.name,
                )
            )

        return results

    def _generate_candidates(self, text: str) -> List[tuple[int, int]]:
        """Collect token spans likely to contain PII.

        This method deliberately over-collects *candidates* but downstream
        similarity + thresholds + length guards keep final spans conservative.
        """
        search_text = normalize_text(text)
        # If NFKC normalization changed length, fall back to original text to
        # avoid misaligned offsets.
        if len(search_text) != len(text):
            search_text = text

        candidates: List[tuple[int, int]] = []

        # Token-based candidates: only available if we have a fast tokenizer.
        if self.tokenizer is not None and getattr(self.tokenizer, "is_fast", False):
            encoded = self.tokenizer(
                search_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False,
            )
            offsets: Sequence[tuple[int, int]] = encoded["offset_mapping"]

            token_chunks = split_with_token_budget(
                search_text,
                offsets,
                max_tokens=self.max_tokens,
                stride=self.TOKEN_STRIDE,
            )

            for chunk_start, chunk_end in token_chunks:
                for token_start, token_end in offsets:
                    abs_start = token_start
                    abs_end = token_end
                    if abs_start < chunk_start or abs_start >= chunk_end:
                        continue

                    token_text = search_text[abs_start:abs_end]
                    if not token_text:
                        continue

                    # Only consider tokens that look like PII-bearing tokens.
                    if any(ch.isdigit() for ch in token_text) or "@" in token_text:
                        # Expand slightly to the right to capture small n-grams,
                        # but stay within a safe bound.
                        span_end = min(len(search_text), abs_end + 24)
                        candidates.append((abs_start, span_end))

        # Regex-based fallback candidates to ensure we still work without tokenizer.
        for pattern in CANDIDATE_REGEXES:
            for match in pattern.finditer(search_text):
                candidates.append(match.span())

        # Deduplicate and clamp.
        deduped: List[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        text_len = len(search_text)

        for start, end in candidates:
            start = max(0, start)
            end = min(text_len, end)
            if end <= start:
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((start, end))

        return deduped
