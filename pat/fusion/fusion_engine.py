"""
The FusionEngine merges overlapping spans from different detectors into a
single, consolidated FusedSpan with deterministic boundary and type handling.
"""

from __future__ import annotations

import logging
from typing import Sequence

from pat.config import get_settings
from pat.detectors import DetectorResult
from pat.detectors.debug_logging import log_decision
from pat.fusion.span import FusedSpan
from pat.utils.taxonomy import category_for_type, priority_for_type
from pat.utils.text import trim_span

LOG = logging.getLogger(__name__)


class FusionEngine:
    """Merges overlapping DetectorResult spans into FusedSpan objects."""

    MERGE_GAP = 1
    MIN_IOU_FOR_MISMATCHED_TYPES = 0.1
    SCORE_BONUS_PER_DETECTOR = 0.05
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
        self.detector_weights = settings.detector_weights
        self.context_window = context_window or getattr(settings, "fusion_context_window", 80)
        self.debug_enabled = getattr(settings, "debug_fusion", False)
        try:
            from pat.utils.taxonomy import pii_schema

            merged = dict(self.TYPE_PRIORITY)
            merged.update({k: int(v) for k, v in pii_schema().get("type_priority", {}).items()})
            self.TYPE_PRIORITY = merged
        except Exception:
            ...

    def fuse(self, results: Sequence[DetectorResult], *, text: str | None = None) -> list[FusedSpan]:
        """
        Fuses a list of detector results into a minimal list of FusedSpans.

        Overlap or adjacency is required for a merge to prevent runaway boundary
        creep. When types differ, a minimum IoU gate prevents unrelated spans
        from being merged while still collapsing obviously co-referent spans.
        """
        if not results:
            return []

        # Stable sort: start asc, longer span first, higher confidence first.
        sorted_results = sorted(
            results,
            key=lambda r: (r.start, -(r.end - r.start), -(getattr(r, "score", None) or r.confidence)),
        )

        fused_spans: list[FusedSpan] = []
        current_span = FusedSpan.from_detector_result(sorted_results[0])

        for next_result in sorted_results[1:]:
            if self._should_merge(current_span, next_result):
                self._merge_into(current_span, next_result)
            else:
                self._finalize_span(current_span, text, fused_spans)
                current_span = FusedSpan.from_detector_result(next_result)

        self._finalize_span(current_span, text, fused_spans)
        return fused_spans

    def _should_merge(self, fused: FusedSpan, incoming: DetectorResult) -> bool:
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
        incoming_type = incoming.pii_type
        incoming_cat = getattr(incoming, "category", None) or category_for_type(incoming_type)
        fused_cats = fused.all_categories or ({fused.category} if fused.category else set())
        if incoming_type in fused.all_types:
            return True
        if self._is_name_type(incoming_type) and any(
            self._is_name_type(t) for t in fused.all_types
        ):
            return True
        if self._is_address_type(incoming_type) and any(
            self._is_address_type(t) for t in fused.all_types
        ):
            return True
        if self._is_numeric_type(incoming_type) and any(
            self._is_numeric_type(t) for t in fused.all_types
        ):
            return True

        if allow_adjacent:
            # Allow adjacency merges for obvious continuations like multi-token names/addresses.
            return self._is_name_type(incoming_type) or self._is_address_type(incoming_type)

        if fused_cats and incoming_cat and incoming_cat not in fused_cats:
            return False

        # Only merge conflicting types if there is meaningful overlap.
        union = (fused.end - fused.start) + (incoming.end - incoming.start) - overlap
        iou = overlap / union if union > 0 else 0.0
        return iou >= self.MIN_IOU_FOR_MISMATCHED_TYPES

    def _merge_into(self, fused: FusedSpan, incoming: DetectorResult) -> None:
        before = (fused.start, fused.end, fused.pii_type, float(fused.confidence))
        fused.start = min(fused.start, incoming.start)
        fused.end = max(fused.end, incoming.end)
        fused.add_source(incoming)

        type_scores = self._score_types(fused.sources)
        best_type = max(
            type_scores,
            key=lambda t: (type_scores[t], self.TYPE_PRIORITY.get(t, priority_for_type(t, 0))),
        )
        fused.pii_type = best_type
        fused.category = fused.category or category_for_type(fused.pii_type)
        if fused.category:
            fused.all_categories.add(fused.category)
        fused.confidence = self._aggregate_confidence(fused)
        fused.metadata["all_types"] = sorted(fused.all_types)
        fused.metadata["all_categories"] = sorted(c for c in fused.all_categories if c)
        fused.metadata["detectors"] = sorted(fused.detectors)
        fused.metadata["type_scores"] = type_scores

        if self.debug_enabled:
            log_decision(
                detector_name="fusion",
                action="merge",
                text_window="",
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
        scores: dict[str, float] = {}
        for src in sources:
            weight = self.detector_weights.get(src.detector_name, 1.0)
            score_val = getattr(src, "score", None)
            score = float(score_val if score_val is not None else src.confidence) * weight
            scores[src.pii_type] = max(scores.get(src.pii_type, 0.0), score)
        return scores

    def _aggregate_confidence(self, span: FusedSpan) -> float:
        detector_scores = sorted(span.detector_scores.values(), reverse=True)
        if not detector_scores:
            return 0.0
        base = detector_scores[0]
        bonus = self.SCORE_BONUS_PER_DETECTOR * max(0, len(span.detectors) - 1)
        return min(1.0, base + bonus)

    def _finalize_span(self, span: FusedSpan, text: str | None, bucket: list[FusedSpan]) -> None:
        if text is not None:
            start, end, sliced = trim_span(text, span.start, span.end)
            span.start, span.end, span.text = start, end, sliced
            span.left_context = text[max(0, start - self.context_window) : start]
            span.right_context = text[end : min(len(text), end + self.context_window)]
        bucket.append(span)

    def _is_name_type(self, pii_type: str) -> bool:
        return pii_type in self.NAME_TYPES or pii_type.upper().startswith("PERSON")

    def _is_address_type(self, pii_type: str) -> bool:
        return pii_type in self.ADDRESS_TYPES

    def _is_numeric_type(self, pii_type: str) -> bool:
        return pii_type in self.NUMERIC_TYPES
