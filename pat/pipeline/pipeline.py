"""The core RedactionPipeline orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from pat.detectors.runner import DetectorRunner
from pat.embeddings import EmbeddingModel
from pat.fusion import FusionEngine, FusedSpan
from pat.policy.engine import PolicyDecision, PolicyEngine
from pat.severity.model import SeverityModel
from pat.severity.features import (
    compute_neighbor_stats,
    compute_span_sentence_context,
    compute_token_position_ratio,
    extract_span_features,
    get_span_context_text,
    span_features_to_vector,
)
from pat.utils.text import compute_sentence_boundaries

LOG = logging.getLogger(__name__)


def _apply_redactions(text: str, decisions: list[PolicyDecision]) -> str:
    """Apply redaction decisions to text safely."""
    # Sort decisions by start index, descending, to avoid index shifting issues.
    sorted_decisions = sorted(decisions, key=lambda d: d.span.start, reverse=True)
    sanitized_text = text

    for decision in sorted_decisions:
        span = decision.span
        start, end = span.start, span.end

        if decision.action == "ALLOW":
            continue

        if decision.action == "BLOCK":
            # In a real service, this would raise an exception to be caught by a framework.
            # For this implementation, we'll log and mask.
            LOG.error("BLOCK action triggered by rule %s. Masking content.", decision.rule_id)

        replacement = decision.masked_text or decision.placeholder
        if replacement is None:
            # If a MASK/BLOCK action has no placeholder, it's a policy misconfiguration.
            # We provide a sensible default instead of failing.
            replacement = f"[{span.pii_type}]"
        if start < 0 or end > len(sanitized_text) or start > end:
            LOG.warning("Skipping invalid span application: %s", decision)
            continue
        sanitized_text = sanitized_text[:start] + replacement + sanitized_text[end:]

    return sanitized_text


class RedactionPipeline:
    """Orchestrates the PII detection and redaction pipeline."""

    def __init__(
        self,
        *,
        detector_runner: DetectorRunner | None = None,
        fusion_engine: FusionEngine | None = None,
        severity_model: SeverityModel | None = None,
        policy_engine: PolicyEngine | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        """Initialise the pipeline with its component engines."""
        self.detector_runner = detector_runner or DetectorRunner()
        self.fusion_engine = fusion_engine or FusionEngine()
        self.severity_model = severity_model or SeverityModel()
        self.policy_engine = policy_engine or PolicyEngine()
        self.embedding_model = embedding_model or EmbeddingModel()

    def run(self, text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute the full PII sanitization pipeline on a string of text.

        Args:
            text: The input text to sanitize.
            context: Optional context for policy decisions (e.g., channel).

        Returns:
            A dictionary containing the results of the pipeline run.
        """
        # 1. Run all detectors
        raw_detections = self.detector_runner.run(text)

        # 2. Fuse detector results
        fused_spans = self.fusion_engine.fuse(raw_detections, text=text)

        # 3. Score severity using the trained ML model
        span_severity_scores: list[float] = []
        if fused_spans and getattr(self.severity_model, "model", None):
            sentences = compute_sentence_boundaries(text)
            neighbor_stats = compute_neighbor_stats(fused_spans)

            # Batch encode embeddings for efficiency
            span_texts = [s.text for s in fused_spans]
            context_texts = [get_span_context_text(text, s.start, s.end) for s in fused_spans]
            span_embeddings = self.embedding_model.encode_batch(span_texts)
            context_embeddings = self.embedding_model.encode_batch(context_texts)

            for idx, span in enumerate(fused_spans):
                sentence_index, sentence_ratio = compute_span_sentence_context(span, sentences)
                token_ratio = compute_token_position_ratio(text, span)
                neighbor_span_count, neighbor_high_risk_count = neighbor_stats.get(idx, (0, 0))

                features = extract_span_features(
                    span,
                    text,
                    sentence_index=sentence_index,
                    sentence_position_ratio=sentence_ratio,
                    token_position_ratio=token_ratio,
                    embedding=span_embeddings[idx],
                    context_embedding=context_embeddings[idx],
                    neighbor_span_count=neighbor_span_count,
                    neighbor_high_risk_count=neighbor_high_risk_count,
                )
                feature_vector = span_features_to_vector(features)
                score, label, _ = self.severity_model.predict(feature_vector.tolist())
                span_severity_scores.append(score)
                # Attach severity info directly to the span for richer output
                span.severity_score = score
                span.severity_label = label

        # Conservative fallback when the severity model is unavailable or yields no scores.
        if not span_severity_scores and fused_spans:
            for span in fused_spans:
                fallback_score = getattr(span, "max_confidence", 0.0)
                span.severity_score = fallback_score
                span.severity_label = self.policy_engine.get_severity_label(fallback_score)
                span_severity_scores.append(fallback_score)

        overall_severity_score = max(span_severity_scores) if span_severity_scores else 0.0

        severity_label = self.policy_engine.get_severity_label(overall_severity_score)

        # 4. Make policy decisions
        decisions = self.policy_engine.decide(fused_spans, context or {})

        # 5. Apply redactions to produce sanitized text
        sanitized_text = _apply_redactions(text, decisions)

        return {
            "sanitized_text": sanitized_text,
            "decision": "BLOCK" if any(d.action == "BLOCK" for d in decisions) else "PROCESS",
            "severity_label": severity_label,
            "severity_score": overall_severity_score,
            "pii_spans": fused_spans,
            "raw_detector_results": raw_detections,
            "applied_rules": [d.rule_id for d in decisions if d.rule_id],
        }
