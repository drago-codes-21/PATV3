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
    """Apply redaction decisions to text in a single, offset-safe pass."""

    def _severity_rank(label: str | None) -> int:
        order = {"VERY_HIGH": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
        return order.get(label or "", 1)

    actionable = [
        d for d in decisions if d.action in {"MASK", "BLOCK"} and d.span.start < d.span.end
    ]
    if not actionable:
        return text

    # Sort by start position ascending for deterministic left-to-right construction.
    actionable.sort(key=lambda d: (d.span.start, d.span.end))

    instructions: list[dict[str, object]] = []
    for decision in actionable:
        span = decision.span
        start, end = span.start, span.end
        if start >= len(text):
            continue
        end = min(end, len(text))
        if start >= end:
            continue

        replacement = decision.masked_text or decision.placeholder
        if replacement is None:
            replacement = f"<{span.pii_type}>"

        rank = (
            _severity_rank(decision.severity_label or getattr(span, "severity_label", None)),
            1 if decision.action == "MASK" else 2,  # BLOCK outranks MASK if present
            end - start,
            float(getattr(span, "confidence", 0.0) or 0.0),
        )

        instr = {"start": start, "end": end, "replacement": replacement, "rank": rank}

        if not instructions:
            instructions.append(instr)
            continue

        last = instructions[-1]
        last_start, last_end = int(last["start"]), int(last["end"])

        if start >= last_end:
            instructions.append(instr)
            continue

        # Overlap handling: prefer higher-ranked decision, but preserve coverage.
        if rank > last["rank"]:  # type: ignore[index]
            # Preserve the left non-overlapping part of the previous instruction.
            if last_start < start:
                last["end"] = start
            else:
                instructions.pop()
            instructions.append(instr)
        else:
            # Keep existing instruction; if incoming extends beyond last_end, cover the tail.
            if end > last_end:
                tail = dict(instr)
                tail["start"] = last_end
                tail["end"] = end
                instructions.append(tail)

    # Drop any zero-length artifacts from overlap trimming.
    instructions = [i for i in instructions if int(i["end"]) > int(i["start"])]

    # Build sanitized text in a single pass.
    sanitized_parts: list[str] = []
    cursor = 0
    for instr in instructions:
        start, end, replacement = int(instr["start"]), int(instr["end"]), str(instr["replacement"])
        if start < cursor:
            continue  # Defensive guard against malformed overlaps

        # Skip re-masking already-placeholder text.
        segment = text[start:end]
        segment_stripped = segment.strip()
        if segment_stripped.startswith("<") and segment_stripped.endswith(">") and segment_stripped == replacement:
            cursor = end
            continue

        sanitized_parts.append(text[cursor:start])
        sanitized_parts.append(replacement)
        cursor = end

    sanitized_parts.append(text[cursor:])
    return "".join(sanitized_parts)


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
