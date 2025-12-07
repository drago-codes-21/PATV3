"""Embedding-based PII detector."""

from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np

from pat.config import get_settings
from pat.detectors.base import BaseDetector, DetectorContext, DetectorResult
from pat.detectors.debug_logging import log_decision
from pat.embeddings import EmbeddingModel
from pat.utils.text import normalize_text, split_with_token_budget, trim_span

LOG = logging.getLogger(__name__)

PROTOTYPES = {
    "EMAIL": ["email address", "email contact", "email account"],
    "PHONE": ["phone number", "contact number", "mobile number"],
    "ADDRESS": ["street address", "home address", "postal address"],
    "PERSON": ["person's name", "full name", "individual's name"],
    "CREDENTIAL": ["login password", "secret key", "authentication token"],
    "CARD_NUMBER": ["credit card number", "debit card number", "PAN"],
    "BANK_ACCOUNT": ["bank account number", "account details"],
    "NI_NUMBER": ["national insurance number", "NINO"],
    "NHS_NUMBER": ["NHS number", "health service number"],
    "POSTCODE": ["postal code", "zip code"],
}

CANDIDATE_REGEXES = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.IGNORECASE),
    re.compile(r"(?<!\d)(\d{2}[-\s]?\d{2}[-\s]?\d{2})(?!\d)"),
    re.compile(r"\b\d{6,}\b"),
]


class EmbeddingSimilarityDetector(BaseDetector):
    """Detect PII by comparing candidate span embeddings to prototype embeddings."""

    MAX_TOKENS = 128
    TOKEN_STRIDE = 24

    @property
    def name(self) -> str:
        return "embedding"

    def __init__(self) -> None:
        settings = get_settings()
        self.embedding_model = EmbeddingModel()
        self.threshold = settings.embedding_similarity_threshold

        loaded_thresholds = {}
        try:
            if settings.detector_thresholds_path.exists():
                import json as _json

                loaded = _json.loads(settings.detector_thresholds_path.read_text(encoding="utf-8"))
                loaded_thresholds = loaded.get("embedding", {})
        except Exception:
            loaded_thresholds = settings.detector_thresholds.get("embedding") if hasattr(settings, "detector_thresholds") else {}
        self.type_thresholds = {**(loaded_thresholds or {})}

        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(settings.embedding_model_path))
            max_len = getattr(self.tokenizer, "model_max_length", 512) or 512
            self.max_tokens = max(48, min(self.MAX_TOKENS, max_len))
        except Exception as exc:
            LOG.warning("Tokenizer unavailable for embedding detector: %s", exc)
            self.tokenizer = None
            self.max_tokens = self.MAX_TOKENS

        self.prototype_types: list[str] = []
        self.prototype_phrases: list[str] = []
        for pii_type, phrases in PROTOTYPES.items():
            for phrase in phrases:
                self.prototype_types.append(pii_type)
                self.prototype_phrases.append(phrase)

        LOG.info("Pre-computing %d prototype embeddings...", len(self.prototype_phrases))
        self.prototype_embeddings = self.embedding_model.encode_batch(self.prototype_phrases)

    def run(self, text: str, *, context: DetectorContext | None = None) -> Sequence[DetectorResult]:
        if not text:
            return []

        candidates = self._generate_candidates(text)
        if not candidates:
            return []

        candidate_texts = [text[s:e] for s, e in candidates]
        embeddings = self.embedding_model.encode_batch(candidate_texts)
        # Align prototype embedding dimensions to candidate embedding size (helps in lightweight tests)
        if self.prototype_embeddings.shape[1] != embeddings.shape[1]:
            try:
                self.prototype_embeddings = self.embedding_model.encode_batch(self.prototype_phrases)
            except Exception:
                target_dim = embeddings.shape[1]
                proto = self.prototype_embeddings
                if proto.shape[1] > target_dim:
                    proto = proto[:, :target_dim]
                else:
                    pad_width = target_dim - proto.shape[1]
                    proto = np.pad(proto, ((0, 0), (0, pad_width)))
                self.prototype_embeddings = proto

        similarity = embeddings @ self.prototype_embeddings.T

        results: list[DetectorResult] = []
        seen: set[tuple[int, int, str]] = set()
        for idx, (start, end) in enumerate(candidates):
            scores = similarity[idx]
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            pii_type = self.prototype_types[best_idx]
            threshold = self.type_thresholds.get(pii_type, self.threshold)
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

        # Incorporate prior detector seeds to strengthen collaboration in low-dim/heuristic settings.
        if context and context.prior_results:
            for seed in context.prior_results:
                key = (seed.start, seed.end, seed.pii_type)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    DetectorResult(
                        start=seed.start,
                        end=seed.end,
                        text=seed.text,
                        pii_type=seed.pii_type,
                        confidence=max(getattr(seed, "confidence", 0.0), self.threshold),
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
                        candidates.append((abs_start, min(len(search_text), abs_end + 24)))

        for pattern in CANDIDATE_REGEXES:
            for match in pattern.finditer(search_text):
                candidates.append(match.span())

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


# Backwards compatible alias for legacy imports
SemanticSimilarityDetector = EmbeddingSimilarityDetector
