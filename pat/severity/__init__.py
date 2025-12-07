"""Severity scoring utilities."""

from .features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    compute_neighbor_stats,
    compute_span_sentence_context,
    compute_token_position_ratio,
    extract_span_features,
    get_span_context_text,
    span_features_to_vector,
)
from .model import SeverityModel

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "SeverityModel",
    "compute_neighbor_stats",
    "compute_span_sentence_context",
    "compute_token_position_ratio",
    "extract_span_features",
    "get_span_context_text",
    "span_features_to_vector",
]
