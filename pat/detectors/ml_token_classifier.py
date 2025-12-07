"""Token-level classifier backed by mpnet embeddings."""

from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np

from .base import BaseDetector, DetectorContext, DetectorResult
from .debug_logging import log_decision
from pat.config import get_settings
from pat.embeddings import EmbeddingModel
from pat.utils.text import normalize_text, split_with_token_budget, trim_span

LOG = logging.getLogger(__name__)

TOKEN_PROTOTYPES = {
    "CARD_NUMBER": ["16 digit card number", "long card number", "credit card digits"],
    "BANK_ACCOUNT": ["bank account digits", "eight digit account number"],
    "SORT_CODE": ["bank sort code", "sort code digits"],
    "PHONE": ["mobile phone digits", "phone contact number"],
    "EMAIL": ["email address token", "email handle"],
    "ADDRESS": ["street and house number", "address line"],
    "POSTCODE": ["uk postcode token", "postal code"],
    "NI_NUMBER": ["national insurance number", "ni number"],
    "CREDENTIAL": ["one time password code", "temporary passcode", "login passcode"],
}

CANDIDATE_REGEXES = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.IGNORECASE),
    re.compile(r"(?<!\d)(\d{2}[-\s]?\d{2}[-\s]?\d{2})(?!\d)"),
    re.compile(r"\b\d{6,}\b"),
]


class MLTokenClassifierDetector(BaseDetector):
    """Classify token-level spans using mpnet embeddings."""

    MAX_TOKENS = 128
    TOKEN_STRIDE = 24
    DEFAULT_THRESHOLD = 0.48
    TYPE_THRESHOLDS = {
        "CARD_NUMBER": 0.55,
        "BANK_ACCOUNT": 0.5,
        "SORT_CODE": 0.5,
        "CREDENTIAL": 0.48,
        "PHONE": 0.5,
        "EMAIL": 0.52,
        "POSTCODE": 0.48,
        "NI_NUMBER": 0.5,
        "ADDRESS": 0.45,
    }

    @property
    def name(self) -> str:
        return "ml_token"

    def __init__(self) -> None:
        settings = get_settings()
        self.embedding_model = EmbeddingModel()
        loaded_thresholds = {}
        try:
            if settings.detector_thresholds_path.exists():
                import json as _json

                loaded = _json.loads(settings.detector_thresholds_path.read_text(encoding="utf-8"))
                loaded_thresholds = loaded.get("ml_token", {})
        except Exception:
            loaded_thresholds = settings.detector_thresholds.get("ml_token") if hasattr(settings, "detector_thresholds") else {}
        self.thresholds = {**self.TYPE_THRESHOLDS, **(loaded_thresholds or {})}

        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(settings.embedding_model_path))
            max_len = getattr(self.tokenizer, "model_max_length", 512) or 512
            self.max_tokens = max(48, min(self.MAX_TOKENS, max_len))
        except Exception as exc:  # pragma: no cover - optional dependency
            LOG.warning("Tokenizer unavailable for ml_token detector: %s", exc)
            self.tokenizer = None
            self.max_tokens = self.MAX_TOKENS

        self.prototype_types: list[str] = []
        prototype_phrases: list[str] = []
        for pii_type, phrases in TOKEN_PROTOTYPES.items():
            for phrase in phrases:
                self.prototype_types.append(pii_type)
                prototype_phrases.append(phrase)

        LOG.info("Pre-computing %d token prototype embeddings...", len(prototype_phrases))
        self.prototype_embeddings = self.embedding_model.encode_batch(prototype_phrases)

    def run(self, text: str, *, context: DetectorContext | None = None) -> Sequence[DetectorResult]:
        if not text:
            return []

        candidates = self._generate_candidates(text)
        if not candidates:
            return []

        candidate_texts = [text[s:e] for s, e in candidates]
        embeddings = self.embedding_model.encode_batch(candidate_texts)
        similarity = embeddings @ self.prototype_embeddings.T

        results: list[DetectorResult] = []
        seen: set[tuple[int, int, str]] = set()
        for idx, (start, end) in enumerate(candidates):
            scores = similarity[idx]
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            pii_type = self.prototype_types[best_idx]
            threshold = self.thresholds.get(pii_type, self.DEFAULT_THRESHOLD)
            if abs(best_score - threshold) <= 0.05:
                log_decision(
                    detector_name=self.name,
                    action="near_threshold",
                    text_window=text[start:end][:160],
                    span_before=(start, end),
                    score=best_score,
                    details={"pii_type": pii_type, "threshold": threshold},
                )
            if best_score < threshold:
                continue
            adj_start, adj_end, adj_text = trim_span(text, start, end)
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

    def _generate_candidates(self, text: str) -> list[tuple[int, int]]:
        """Collect token spans likely to contain PII."""

        search_text = normalize_text(text)
        if len(search_text) != len(text):
            search_text = text

        candidates: list[tuple[int, int]] = []

        if self.tokenizer and getattr(self.tokenizer, "is_fast", False):
            encoded = self.tokenizer(
                search_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False,
            )
            offsets = encoded["offset_mapping"]
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
                    if not (chunk_start <= abs_start < chunk_end):
                        continue
                    token_text = search_text[abs_start:abs_end]
                    if not token_text:
                        continue
                    if any(ch.isdigit() for ch in token_text) or "@" in token_text:
                        # Expand to a small n-gram around the token to capture context.
                        candidates.append((abs_start, min(len(search_text), abs_end + 24)))
        # Regex-based fallback candidates to ensure coverage without tokenizer.
        for pattern in CANDIDATE_REGEXES:
            for match in pattern.finditer(search_text):
                candidates.append(match.span())

        # Deduplicate and clamp.
        deduped: list[tuple[int, int]] = []
        seen = set()
        for start, end in candidates:
            start = max(0, start)
            end = min(len(search_text), end)
            if end <= start:
                continue
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((start, end))
        return deduped
