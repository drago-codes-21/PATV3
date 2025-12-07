"""Span-level feature extraction utilities."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np

from pat.fusion import FusedSpan
from pat.utils.text import compute_sentence_boundaries
from pat.utils.taxonomy import categories_for_types, category_for_type

# ---- Canonical feature schema (order matters!) ----
SOURCE_FEATURES = {
    "regex": "source_regex",
    "ner": "source_ner",
    "ml_ner": "source_ner",
    "embedding": "source_embedding",
    "semantic": "source_embedding",
    "domain": "source_domain_heuristics",
    "domain_heuristic": "source_domain_heuristics",
    "ml_token": "source_ml_token",
}

ALL_PII_TYPES = [
    "BANK_ACCOUNT",
    "SORT_CODE",
    "IBAN",
    "CARD_NUMBER",
    "CUSTOMER_ID",
    "CREDENTIAL",
    "PASSWORD",
    "PIN",
    "TOKEN",
    "API_KEY",
    "SESSION_ID",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "POSTCODE",
    "URL",
    "SOCIAL_HANDLE",
    "NI_NUMBER",
    "PASSPORT_NUMBER",
    "DRIVING_LICENSE",
    "NHS_NUMBER",
    "ORGANIZATION",
    "DATE",
    "MONEY",
    "PERSON",
    "GOV_ID",
    "MEDICAL_INFO",
    "MEDICATION",
    "DIAGNOSIS",
    "HOSPITAL",
    "HEALTH_CONTEXT",
    "IP_ADDRESS",
    "IPV6_ADDRESS",
    "MAC_ADDRESS",
    "DEVICE_ID",
    "COOKIE_ID",
    "TRACKING_ID",
    "EMPLOYER_NAME",
    "JOB_TITLE",
    "EMPLOYMENT_ID",
    "STUDENT_ID",
    "DEGREE",
    "GEO_COORDINATES",
    "TRAVEL_PLAN",
    "EVENT_CHECKIN",
    "ACTIVITY_PATTERN",
    "GENERIC_NUMBER",
]

CATEGORIES = [
    "GOVERNMENT_ID",
    "CONTACT",
    "FINANCIAL",
    "BIOMETRIC",
    "AUTHENTICATION",
    "HEALTH",
    "ONLINE_DEVICE",
    "EMPLOYMENT_EDUCATION",
    "LOCATION_ACTIVITY",
]

CONTEXT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ctx_has_bank": ("bank", "branch", "ifsc"),
    "ctx_has_account": ("account", "acct", "iban"),
    "ctx_has_password": ("password", "passcode", "pwd"),
    "ctx_has_secret": ("secret", "token", "api key", "api-key", "bearer"),
    "ctx_has_otp": ("otp", "one time", "verification code"),
    "ctx_has_confidential": ("confidential", "sensitive", "private"),
    "ctx_has_credit": ("card", "credit", "debit"),
    "ctx_has_address": ("address", "street", "road", "avenue"),
    "ctx_has_phone": ("phone", "mobile", "call"),
    "ctx_has_email": ("email", "mail", "@"),
    "ctx_has_health": ("diagnosis", "treatment", "prescribed", "symptom", "nhs"),
    "ctx_has_employment": ("employer", "manager", "company", "payroll", "job title"),
    "ctx_has_biometric": ("fingerprint", "voiceprint", "face id", "iris"),
    "ctx_has_location": ("gps", "coordinates", "itinerary", "travel", "check-in"),
}

CHANNEL_FEATURES = (
    "context_channel_email",
    "context_channel_chat",
    "context_channel_log",
    "context_channel_other",
    "context_section_subject",
    "context_section_body",
    "context_section_signature",
    "context_section_header",
)

BASE_FEATURES: list[str] = [
    "source_regex",
    "source_ner",
    "source_embedding",
    "source_domain_heuristics",
    "source_ml_token",
    "num_detectors",
    "confidence",
    "fused_score",
    "max_detector_score",
    "mean_detector_score",
    "min_detector_score",
    "regex_score",
    "ner_score",
    "ml_ner_score",
    "embedding_score",
    "semantic_score",
    "domain_score",
    "domain_heuristic_score",
    "ml_token_score",
    "digit_count",
    "alpha_count",
    "special_count",
    "whitespace_count",
    "span_length",
    "digit_ratio",
    "alpha_ratio",
    "special_ratio",
    "whitespace_ratio",
    "has_hyphen",
    "has_plus",
    "has_at",
    "has_special_char",
    "has_punctuation",
    "is_all_digits",
    "is_mixed_case",
    "has_domain_suffix",
    "segment_count",
    "embedding_mean",
    "embedding_std",
    "embedding_norm",
    "embedding_max",
    "is_financial",
    "is_credential",
    "is_contact_detail",
    "is_location_related",
    "sentence_index",
    "sentence_position_ratio",
    "token_position_ratio",
    "context_embedding_norm",
    "span_context_similarity",
    "neighbor_span_count",
    "neighbor_high_risk_count",
] + list(CONTEXT_KEYWORDS.keys()) + list(CHANNEL_FEATURES)

FEATURE_NAMES: list[str] = BASE_FEATURES + [f"pii_type_{t}" for t in ALL_PII_TYPES]
FEATURE_NAMES += [f"pii_category_{c}" for c in CATEGORIES]
FEATURE_SCHEMA_VERSION = "severity_v1"
FEATURE_SCHEMA = {"version": FEATURE_SCHEMA_VERSION, "feature_names": FEATURE_NAMES}
_EXPECTED_FEATURE_COUNT = len(FEATURE_NAMES)
FEATURE_NAME_SET = tuple(FEATURE_NAMES)

FINANCIAL_TYPES = {"BANK_ACCOUNT", "SORT_CODE", "IBAN", "CARD_NUMBER", "CUSTOMER_ID", "TRANSACTION_ID"}
CREDENTIAL_TYPES = {"CREDENTIAL", "PASSWORD", "PIN", "TOKEN", "API_KEY", "SESSION_ID"}
CONTACT_TYPES = {"EMAIL", "PHONE", "ADDRESS", "POSTCODE", "URL", "SOCIAL_HANDLE"}
LOCATION_TYPES = {"ADDRESS", "POSTCODE", "GEO_COORDINATES", "TRAVEL_PLAN", "EVENT_CHECKIN", "ACTIVITY_PATTERN"}
HIGH_RISK_TYPES = FINANCIAL_TYPES | CREDENTIAL_TYPES

CONTEXT_WINDOW_SIZE = 10  # Number of tokens before/after span


def _safe_get_context(span: FusedSpan, text: str, window: int = 80) -> str:
    if getattr(span, "left_context", "") or getattr(span, "right_context", ""):
        return f"{getattr(span, 'left_context', '')} {getattr(span, 'right_context', '')}".strip()
    start = max(0, span.start - window)
    end = min(len(text), span.end + window)
    return text[start:end]


def _keyword_flags(context: str) -> dict[str, float]:
    lowered = context.lower()
    flags: dict[str, float] = {}
    for feature_name, keywords in CONTEXT_KEYWORDS.items():
        flags[feature_name] = 1.0 if any(k in lowered for k in keywords) else 0.0
    return flags


def _channel_flags(context_metadata: Mapping[str, Any] | None) -> dict[str, float]:
    flags = {name: 0.0 for name in CHANNEL_FEATURES}
    if not context_metadata:
        return flags
    channel = str(context_metadata.get("channel", "")).lower()
    section = str(context_metadata.get("section", "")).lower()
    if "email" in channel:
        flags["context_channel_email"] = 1.0
    elif "chat" in channel or "sms" in channel:
        flags["context_channel_chat"] = 1.0
    elif "log" in channel:
        flags["context_channel_log"] = 1.0
    elif channel:
        flags["context_channel_other"] = 1.0
    if "subject" in section:
        flags["context_section_subject"] = 1.0
    elif "signature" in section:
        flags["context_section_signature"] = 1.0
    elif "header" in section:
        flags["context_section_header"] = 1.0
    elif section:
        flags["context_section_body"] = 1.0
    return flags


def extract_span_features(
    span: FusedSpan,
    text: str,
    *,
    sentence_index: int | None = None,
    sentence_position_ratio: float | None = None,
    token_position_ratio: float | None = None,
    embedding: np.ndarray | None = None,
    domain_flags: dict[str, bool] | None = None,
    context_embedding: np.ndarray | None = None,
    neighbor_span_count: int = 0,
    neighbor_high_risk_count: int = 0,
    context_metadata: Mapping[str, Any] | None = None,
    keyword_flags: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Extract per-span feature dictionary."""

    features: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
    span_text = span.text or text[span.start : span.end]
    span_length = len(span_text)

    # Normalise detectors/sources to string names
    raw_sources = list(getattr(span, "detectors", [])) or list(getattr(span, "sources", []) or [])
    detectors = []
    for src in raw_sources:
        if isinstance(src, str):
            detectors.append(src)
        elif hasattr(src, "detector_name"):
            detectors.append(getattr(src, "detector_name"))
        else:
            detectors.append(str(src))

    # The individual detector scores are lost in fusion; we can only create a map using the max_confidence.
    scores_map = getattr(span, "detector_scores", {}) or {name: span.max_confidence for name in detectors}
    detector_scores = list(scores_map.values()) or [span.max_confidence]
    for source_name in detectors:
        feature_name = SOURCE_FEATURES.get(source_name, "")
        if feature_name in features:
            features[feature_name] = 1.0
    features["num_detectors"] = float(len(set(detectors)))
    features["confidence"] = float(span.max_confidence)
    features["fused_score"] = float(span.max_confidence)
    if detector_scores:
        features["max_detector_score"] = float(max(detector_scores))
        features["mean_detector_score"] = float(sum(detector_scores) / max(1, len(detector_scores)))
        features["min_detector_score"] = float(min(detector_scores))
    for det_name, score in scores_map.items():
        score_name = f"{det_name}_score"
        if score_name in features:
            features[score_name] = float(score)

    digit_count = sum(char.isdigit() for char in span_text)
    alpha_count = sum(char.isalpha() for char in span_text)
    special_count = sum((not c.isalnum()) and not c.isspace() for c in span_text)
    whitespace_count = sum(c.isspace() for c in span_text)
    features["digit_count"] = float(digit_count)
    features["alpha_count"] = float(alpha_count)
    features["special_count"] = float(special_count)
    features["whitespace_count"] = float(whitespace_count)
    features["span_length"] = float(span_length)
    if span_length > 0:
        features["digit_ratio"] = digit_count / span_length
        features["alpha_ratio"] = alpha_count / span_length
        features["special_ratio"] = special_count / span_length
        features["whitespace_ratio"] = whitespace_count / span_length

    features["has_hyphen"] = 1.0 if "-" in span_text else 0.0
    features["has_plus"] = 1.0 if "+" in span_text else 0.0
    features["has_at"] = 1.0 if "@" in span_text else 0.0
    features["has_special_char"] = 1.0 if special_count > 0 else 0.0
    features["has_punctuation"] = 1.0 if bool(re.search(r"[.,;:!?]", span_text)) else 0.0
    features["is_all_digits"] = 1.0 if digit_count == span_length and span_length > 0 else 0.0
    features["is_mixed_case"] = 1.0 if any(c.islower() for c in span_text) and any(c.isupper() for c in span_text) else 0.0
    features["has_domain_suffix"] = 1.0 if re.search(r"\.[a-z]{2,}$", span_text.lower()) else 0.0
    segments = [seg for seg in re.split(r"[\s,/;:]+", span_text) if seg]
    features["segment_count"] = float(len(segments))

    vector = embedding if embedding is not None else np.zeros(1, dtype=float)
    if vector.size:
        features["embedding_mean"] = float(np.mean(vector))
        features["embedding_std"] = float(np.std(vector))
        features["embedding_norm"] = float(np.linalg.norm(vector))
        features["embedding_max"] = float(np.max(vector))

    type_set = set(getattr(span, "all_types", []))
    type_set.add(span.pii_type)
    # Individual source PII types are lost during fusion. The FusedSpan only retains the best one.
    # The logic below correctly uses the available type information.
    features["is_credential"] = 1.0 if type_set & CREDENTIAL_TYPES else 0.0
    features["is_contact_detail"] = 1.0 if type_set & CONTACT_TYPES else 0.0
    features["is_location_related"] = 1.0 if type_set & LOCATION_TYPES else 0.0

    # Add one-hot encoded PII types across fused span and all sources.
    for pii_type in type_set:
        pii_type_feature = f"pii_type_{pii_type.strip()}"
        if pii_type_feature in features:
            features[pii_type_feature] = 1.0

    category_set = categories_for_types(type_set)
    if getattr(span, "category", None):
        category_set.add(span.category)
    for cat in category_set:
        feat = f"pii_category_{cat}"
        if feat in features:
            features[feat] = 1.0

    if domain_flags:
        for key in ("is_financial", "is_credential", "is_contact_detail", "is_location_related"):
            if domain_flags.get(key):
                features[key] = 1.0

    if sentence_index is not None and sentence_position_ratio is not None:
        features["sentence_index"] = float(sentence_index)
        features["sentence_position_ratio"] = sentence_position_ratio
    else:
        sentences = compute_sentence_boundaries(text)
        sent_idx, sent_ratio = compute_span_sentence_context(span, sentences)
        features["sentence_index"] = float(sent_idx)
        features["sentence_position_ratio"] = sent_ratio

    if token_position_ratio is not None:
        features["token_position_ratio"] = token_position_ratio
    else:
        features["token_position_ratio"] = compute_token_position_ratio(text, span)

    # Neighbor density
    features["neighbor_span_count"] = float(max(0, neighbor_span_count))
    features["neighbor_high_risk_count"] = float(max(0, neighbor_high_risk_count))

    # Add features from context embedding
    context_vec = context_embedding if context_embedding is not None else np.zeros(1, dtype=float)
    if context_vec.size:
        features["context_embedding_norm"] = float(np.linalg.norm(context_vec))

    # Add cosine similarity between span and context embeddings
    if vector.size and context_vec.size and np.any(vector) and np.any(context_vec):
        sim = np.dot(vector, context_vec.T) / (
            np.linalg.norm(vector) * np.linalg.norm(context_vec)
        )
        features["span_context_similarity"] = float(sim)

    # Context keyword and channel features
    context_text = _safe_get_context(span, text)
    flags = keyword_flags or _keyword_flags(context_text)
    for key, value in flags.items():
        if key in features:
            features[key] = float(value)
    for key, value in _channel_flags(context_metadata).items():
        if key in features:
            features[key] = float(value)

    return features


def compute_neighbor_stats(
    spans: Sequence[FusedSpan], *, window_chars: int = 64
) -> dict[int, tuple[int, int]]:
    """Compute neighbor span and high-risk counts for each span index."""

    stats: dict[int, tuple[int, int]] = {}
    for idx, span in enumerate(spans):
        count = 0
        high_risk = 0
        for jdx, other in enumerate(spans):
            if idx == jdx:
                continue
            if abs(other.start - span.start) <= window_chars or abs(other.end - span.end) <= window_chars:
                count += 1
                if set(getattr(other, "all_types", [])) & HIGH_RISK_TYPES or other.pii_type in HIGH_RISK_TYPES:
                    high_risk += 1
        stats[idx] = (count, high_risk)
    return stats


def compute_span_sentence_context(
    span: FusedSpan, sentences: Sequence[tuple[int, int]]
) -> tuple[int, float]:
    """Finds the sentence index and relative position for a span."""
    for idx, (start, end) in enumerate(sentences):
        if start <= span.start < end:
            sentence_length = end - start
            relative_position = (span.start - start) / max(1, sentence_length)
            return idx, relative_position
    return 0, 0.0


def compute_token_position_ratio(text: str, span: FusedSpan) -> float:
    """Computes the relative token position of a span in the text."""
    if not text:
        return 0.0
    before_span_text = text[: span.start]
    tokens_before = len(before_span_text.split())
    total_tokens = len(text.split())
    if total_tokens == 0:
        return 0.0
    return min(1.0, tokens_before / total_tokens)


def get_span_context_text(
    text: str, span_start: int, span_end: int, window_size: int = CONTEXT_WINDOW_SIZE
) -> str:
    """Extracts the surrounding text of a span based on a token window."""
    if not text:
        return ""

    pre_context_tokens = text[:span_start].split()
    post_context_tokens = text[span_end:].split()

    context_tokens = pre_context_tokens[-window_size:] + post_context_tokens[:window_size]
    return " ".join(context_tokens)


def span_features_to_vector(features: dict[str, float]) -> np.ndarray:
    """Convert ordered feature dict to numpy vector."""

    missing = set(FEATURE_NAMES) - set(features.keys())
    extra = set(features.keys()) - set(FEATURE_NAMES)
    if missing:
        raise AssertionError(f"Missing feature keys: {sorted(missing)}")
    if extra:
        raise AssertionError(f"Unexpected feature keys: {sorted(extra)}")

    vector = np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=float)
    if vector.shape[0] != _EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"Feature vector length mismatch: expected {_EXPECTED_FEATURE_COUNT}, got {vector.shape[0]}"
        )
    return vector


def assert_feature_schema(names: Sequence[str]) -> None:
    """Raise AssertionError if the provided names diverge from the canonical schema."""

    if len(names) != _EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"Feature schema length mismatch: expected {_EXPECTED_FEATURE_COUNT}, got {len(names)}"
        )
    if list(names) != FEATURE_NAMES:
        raise AssertionError(
            f"Feature schema order/content mismatch against canonical FEATURE_NAMES (schema {FEATURE_SCHEMA_VERSION})."
        )
