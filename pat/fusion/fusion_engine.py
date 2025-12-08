"""
The FusionEngine merges overlapping spans from different detectors into a
single, consolidated FusedSpan with deterministic boundary and type handling.
"""

from __future__ import annotations

import logging
from typing import Sequence, List

from pat.config import get_settings
from pat.detectors.base import DetectorResult
from pat.detectors.debug_logging import log_decision
from pat.fusion.span import FusedSpan
from pat.utils.taxonomy import category_for_type, priority_for_type
from pat.utils.text import trim_span

LOG = logging.getLogger(__name__)


class FusionEngine:
    """Merges overlapping DetectorResult spans into FusedSpan objects."""

    # Max distance (in characters) to treat two spans as "adjacent" for merging
    MERGE_GAP = 1

    # IoU threshold for merging spans with different types but strong overlap
    MIN_IOU_FOR_MISMATCHED_TYPES = 0.1

    # Bonus added to confidence for each additional agreeing detector
    SCORE_BONUS_PER_DETECTOR = 0.05

    # Base type priority; can be extended from taxonomy
    TYPE_PRIORITY: dict[str, int] = {
        "CREDENTIAL": 90,
        "CARD_NUMBER": 85,
        "BANK_ACCOUNT": 82,
        "SORT_CODE": 80,
        "CUSTOMER_ID": 78,
        "GOV_ID": 75,
        "EMAIL": 72,
        "PHONE": 70,
        "ADDRESS": 65,
        "POSTCODE": 64,
        "IP_ADDRESS": 60,
        "URL": 55,
        "PERSON": 52,
        "ORGANIZATION": 50,
        "DATE": 40,
        "MONEY": 38,
        "GENERIC_NUMBER": 15,
    }

    NAME_TYPES = {"PERSON", "PERSON_NAME"}
    ADDRESS_TYPES = {"ADDRESS", "LOCATION", "LOC", "GPE", "POSTCODE"}
    NUMERIC_TYPES = {
        "BANK_ACCOUNT",
        "CARD_NUMBER",
        "SORT_CODE",
        "CUSTOMER_ID",
        "GOV_ID",
        "GENERIC_NUMBER",
        "PHONE",
    }

    def __init__(self, *, context_window: int | None = None) -> None:
        settings = get_settings()
        self.detector_weights = getattr(settings, "detector_weights", {}) or {}
        self.context_window = context_window or getattr(settings, "fusion_context_window", 80)
        self.debug_enabled = getattr(settings, "debug_fusion", False)

        # Allow taxonomy to extend/override priority
        try:
            from pat.utils.taxonomy import pii_schema

            schema = pii_schema() or {}
            extra_priorities = {
                k: int(v) for k, v in schema.get("type_priority", {}).items()
            }
            merged = dict(self.TYPE_PRIORITY)
            merged.update(extra_priorities)
            self.TYPE_PRIORITY = merged
        except Exception as e:
            LOG.warning(f"Failed to load type_priority from taxonomy: {e}")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fuse(self, results: Sequence[DetectorResult], *, text: str | None = None) -> list[FusedSpan]:
        """
        Fuses a list of detector results into a minimal list of FusedSpans.

        - Requires overlap or adjacency for merges (prevents runaway boundary creep).
        - Uses type and category compatibility + IoU to avoid unrelated merges.
        """
        if not results:
            return []

        # Filter out obviously invalid spans to avoid corrupt fusion.
        valid_results: List[DetectorResult] = []
        for r in results:
            if r.start >= r.end:
                LOG.warning(
                    f"Skipping invalid DetectorResult with start>=end: "
                    f"start={r.start}, end={r.end}, type={getattr(r, 'pii_type', None)}"
                )
                continue
            valid_results.append(r)

        if not valid_results:
            return []

        # Stable sort: start asc, longer span first, higher confidence/score first.
        def _sort_key(r: DetectorResult) -> tuple[int, int, float]:
            length = r.end - r.start
            score_val = getattr(r, "score", None)
            score = float(score_val) if score_val is not None else float(r.confidence)
            return (r.start, -length, -score)

        sorted_results = sorted(valid_results, key=_sort_key)

        fused_spans: list[FusedSpan] = []
        current_span = FusedSpan.from_detector_result(sorted_results[0])

        for next_result in sorted_results[1:]:
            if self._should_merge(current_span, next_result):
                self._merge_into(current_span, next_result)
            else:
                self._finalize_span(current_span, text, fused_spans)
                current_span = FusedSpan.from_detector_result(next_result)

        self._finalize_span(current_span, text, fused_spans)

        # Ensure deterministic ordering of final spans and drop overlaps defensively.
        fused_spans.sort(key=lambda fs: (fs.start, fs.end))
        fused_spans = self._deduplicate_overlaps(fused_spans)
        return fused_spans

    # ---------------------------------------------------------------------
    # Merge decision logic
    # ---------------------------------------------------------------------
    def _should_merge(self, fused: FusedSpan, incoming: DetectorResult) -> bool:
        """
        Decide whether to merge an incoming detector span into an existing FusedSpan.
        """
        overlap = min(fused.end, incoming.end) - max(fused.start, incoming.start)
        adjacent = incoming.start - fused.end

        if overlap > 0:
            return self._types_compatible(fused, incoming, overlap=overlap)

        if 0 <= adjacent <= self.MERGE_GAP:
            return self._types_compatible(fused, incoming, overlap=0, allow_adjacent=True)

        return False

    def _types_compatible(
        self,
        fused: FusedSpan,
        incoming: DetectorResult,
        *,
        overlap: int,
        allow_adjacent: bool = False,
    ) -> bool:
        """
        Type compatibility heuristic:

        - Always merge if same type.
        - Merge within the same "family" (names, addresses, numeric IDs).
        - For adjacency:
          - Loosen conditions for multi-token names/addresses and numeric fragments.
        - For conflicting categories:
          - Do not merge unless categories agree.
        - For different types but strong overlap:
          - Use IoU threshold to merge obviously co-referent spans.
        """
        incoming_type = incoming.pii_type
        try:
            incoming_cat = getattr(incoming, "category", None) or category_for_type(incoming_type)
        except Exception as e:
            LOG.error(f"Failed to infer category for incoming type {incoming_type}: {e}")
            incoming_cat = None

        fused_cats = fused.all_categories or ({fused.category} if fused.category else set())

        # Same type: always compatible
        if incoming_type in fused.all_types:
            return True

        # Same semantic family (names, addresses, numeric)
        if self._is_name_type(incoming_type) and any(self._is_name_type(t) for t in fused.all_types):
            return True

        if self._is_address_type(incoming_type) and any(
            self._is_address_type(t) for t in fused.all_types
        ):
            return True

        if self._is_numeric_type(incoming_type) and any(
            self._is_numeric_type(t) for t in fused.all_types
        ):
            return True

        # For adjacency, only merge when both spans belong to the same semantic family.
        if allow_adjacent:
            if incoming_type in fused.all_types:
                return True
            if self._is_name_type(incoming_type) and any(self._is_name_type(t) for t in fused.all_types):
                return True
            if self._is_address_type(incoming_type) and any(self._is_address_type(t) for t in fused.all_types):
                return True
            if self._is_numeric_type(incoming_type) and any(
                self._is_numeric_type(t) for t in fused.all_types
            ):
                return True
            return False

        # Categories disagree -> do not merge
        if fused_cats and incoming_cat and incoming_cat not in fused_cats:
            return False

        # Only merge conflicting types if there is meaningful overlap
        fused_len = max(0, fused.end - fused.start)
        incoming_len = max(0, incoming.end - incoming.start)
        union = fused_len + incoming_len - max(0, overlap)
        iou = (overlap / union) if union > 0 else 0.0

        return iou >= self.MIN_IOU_FOR_MISMATCHED_TYPES

    # ---------------------------------------------------------------------
    # Merge & scoring
    # ---------------------------------------------------------------------
    def _merge_into(self, fused: FusedSpan, incoming: DetectorResult) -> None:
        """
        Merge incoming DetectorResult into an existing FusedSpan.
        Expands boundaries, updates type resolution and confidence, and records metadata.
        """
        before = (fused.start, fused.end, fused.pii_type, float(fused.confidence))

        # Expand boundaries to cover both spans
        fused.start = min(fused.start, incoming.start)
        fused.end = max(fused.end, incoming.end)

        # Add evidence
        fused.add_source(incoming)

        # Re-score types based on detector weights and scores
        type_scores = self._score_types(fused.sources)

        # Deterministic type resolution:
        # 1. Max aggregated score
        # 2. Explicit type priority map (or taxonomy priority)
        def _type_rank(t: str) -> tuple[float, int]:
            score = type_scores.get(t, 0.0)
            priority = self.TYPE_PRIORITY.get(t, priority_for_type(t, 0))
            return (score, priority)

        best_type = max(type_scores, key=_type_rank)
        fused.pii_type = best_type

        # Ensure category is synced with final type
        try:
            fused.category = fused.category or category_for_type(fused.pii_type)
        except Exception as e:
            LOG.error(f"Failed to infer category for fused type {fused.pii_type}: {e}")

        if fused.category:
            fused.all_categories.add(fused.category)

        # Aggregate confidence across detectors
        fused.confidence = self._aggregate_confidence(fused)

        if fused.detector_scores:
            fused.max_confidence = max(fused.detector_scores.values())

        # Enrich metadata for downstream introspection/debugging
        fused.metadata["all_types"] = sorted(fused.all_types)
        fused.metadata["all_categories"] = sorted(c for c in fused.all_categories if c)
        fused.metadata["detectors"] = sorted(fused.detectors)
        fused.metadata["type_scores"] = type_scores

        if self.debug_enabled:
            log_decision(
                detector_name="fusion",
                action="merge",
                text_window="",  # can be populated with context if needed
                span_before=(before[0], before[1]),
                span_after=(fused.start, fused.end),
                score=fused.confidence,
                details={
                    "previous_type": before[2],
                    "new_type": fused.pii_type,
                    "types": list(fused.all_types),
                    "detectors": sorted(fused.detectors),
                },
            )

    def _score_types(self, sources: Sequence[DetectorResult]) -> dict[str, float]:
        """
        Compute a score per pii_type based on detector weights and per-detector confidence.
        """
        scores: dict[str, float] = {}
        for src in sources:
            weight = float(self.detector_weights.get(src.detector_name, 1.0))
            score_val = getattr(src, "score", None)
            try:
                base_score = float(score_val) if score_val is not None else float(src.confidence)
            except Exception:
                LOG.warning(
                    f"Non-numeric score/confidence from detector {src.detector_name}; defaulting to 0.0"
                )
                base_score = 0.0

            effective = base_score * weight
            prev = scores.get(src.pii_type, 0.0)
            # Use max to reflect strongest evidence for each type
            scores[src.pii_type] = max(prev, effective)
        return scores

    def _aggregate_confidence(self, span: FusedSpan) -> float:
        """
        Aggregate confidence across detectors for a fused span.
        Uses the strongest detector as base, with a small bonus per additional detector.
        """
        if not span.detector_scores:
            return 0.0

        detector_scores = sorted(span.detector_scores.values(), reverse=True)
        base = detector_scores[0]
        bonus = self.SCORE_BONUS_PER_DETECTOR * max(0, len(span.detectors) - 1)
        return float(min(1.0, max(0.0, base + bonus)))

    # ---------------------------------------------------------------------
    # Finalization & context extraction
    # ---------------------------------------------------------------------
    def _finalize_span(self, span: FusedSpan, text: str | None, bucket: list[FusedSpan]) -> None:
        """
        Finalize a span before returning it:
        - Optionally trim boundaries to token/regex-safe limits.
        - Populate left/right context windows for severity and policy.
        """
        if text is not None:
            start, end, sliced = trim_span(text, span.start, span.end)
            span.start, span.end, span.text = start, end, sliced

            left_start = max(0, start - self.context_window)
            right_end = min(len(text), end + self.context_window)

            span.left_context = text[left_start:start]
            span.right_context = text[end:right_end]

        bucket.append(span)

    def _deduplicate_overlaps(self, spans: list[FusedSpan]) -> list[FusedSpan]:
        """
        Enforce non-overlapping spans. When spans overlap we preserve coverage by
        splitting/triming rather than blindly discarding spans. Preference order:
        higher type priority -> higher confidence -> longer coverage.
        """
        if not spans:
            return []

        resolved: list[FusedSpan] = []

        def _rank(span: FusedSpan) -> tuple[int, float, int]:
            priority = self.TYPE_PRIORITY.get(span.pii_type, priority_for_type(span.pii_type, 0))
            confidence = float(getattr(span, "confidence", 0.0) or 0.0)
            length = max(0, span.end - span.start)
            return (priority, confidence, length)

        for span in spans:
            if not resolved:
                resolved.append(span)
                continue

            last = resolved[-1]
            if span.start >= last.end:
                resolved.append(span)
                continue

            last_rank = _rank(last)
            span_rank = _rank(span)

            # Case 1: incoming span fully contained within last.
            if span.end <= last.end:
                if span_rank > last_rank:
                    # Keep left chunk of last (if any), then the stronger incoming span.
                    if last.start < span.start:
                        resolved[-1] = self._trim_span_copy(last, last.start, span.start) or last
                        resolved.append(span)
                    else:
                        resolved[-1] = span
                # otherwise keep existing last and drop incoming.
                continue

            # Case 2: incoming span extends beyond last.
            if span_rank >= last_rank:
                # Preserve left part of last to avoid gaps, then append stronger span.
                if last.start < span.start:
                    trimmed = self._trim_span_copy(last, last.start, span.start)
                    resolved[-1] = trimmed or last
                else:
                    resolved.pop()
                resolved.append(span)
            else:
                # Keep last, but still cover the tail of the incoming span to avoid missed PII.
                tail = self._trim_span_copy(span, last.end, span.end)
                if tail:
                    resolved.append(tail)

        return resolved

    def _trim_span_copy(self, span: FusedSpan, new_start: int, new_end: int) -> FusedSpan | None:
        """Return a shallow copy of span trimmed to [new_start, new_end)."""
        new_start = max(span.start, new_start)
        new_end = min(span.end, new_end)
        if new_end <= new_start:
            return None

        offset_start = max(0, new_start - span.start)
        offset_end = offset_start + (new_end - new_start)
        trimmed_text = span.text[offset_start:offset_end]

        return FusedSpan(
            start=new_start,
            end=new_end,
            text=trimmed_text,
            pii_type=span.pii_type,
            confidence=span.confidence,
            category=span.category,
            all_types=set(span.all_types),
            all_categories=set(span.all_categories),
            detectors=set(span.detectors),
            detector_scores=dict(span.detector_scores),
            sources=list(span.sources),
            metadata=dict(span.metadata),
            left_context=span.left_context,
            right_context=span.right_context,
            max_confidence=span.max_confidence,
            severity_score=span.severity_score,
            severity_label=span.severity_label,
            severity_probs=span.severity_probs,
        )

    # ---------------------------------------------------------------------
    # Type helpers
    # ---------------------------------------------------------------------
    def _is_name_type(self, pii_type: str) -> bool:
        t = pii_type.upper()
        return t in self.NAME_TYPES or t.startswith("PERSON")

    def _is_address_type(self, pii_type: str) -> bool:
        return pii_type.upper() in self.ADDRESS_TYPES

    def _is_numeric_type(self, pii_type: str) -> bool:
        return pii_type.upper() in self.NUMERIC_TYPES
