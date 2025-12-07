"""Logic for fusing overlapping detections from multiple sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from pat.detectors.base import DetectorResult


@dataclass(frozen=False, unsafe_hash=True)
class FusedSpan:
    """A PII span resulting from the fusion of one or more raw detections."""

    pii_type: str
    text: str
    start: int
    end: int
    sources: Sequence[str]
    max_confidence: float
    validator_passed_all: bool = True
    # Add optional fields for testing convenience
    severity_label: str | None = None
    severity_score: float | None = None


class FusionEngine:
    """Merges and de-duplicates raw detections into fused spans."""

    def fuse(self, spans: list[DetectorResult], text: str) -> list[FusedSpan]:
        """
        Merge overlapping or adjacent spans from different detectors.

        This is a simple greedy implementation that sorts by start position.
        """
        if not spans:
            return []

        # Sort by start position, then by end position descending for stability
        sorted_spans = sorted(spans, key=lambda s: (s.start, -s.end))

        fused: list[FusedSpan] = []
        if not sorted_spans:
            return fused

        current_span_group = [sorted_spans[0]]

        for next_span in sorted_spans[1:]:
            # Check for overlap with the latest span in the current group
            # Allow merging of adjacent or nearly-adjacent spans to handle chunked PII.
            # A small gap (e.g., a space or newline) is allowed.
            if next_span.start <= current_span_group[-1].end + 2:
                current_span_group.append(next_span)
            else:
                # No overlap, finalize the current group and start a new one
                merged = self._merge_span_group(current_span_group, text)
                fused.append(merged)
                current_span_group = [next_span]

        # Add the last processed group
        if current_span_group:
            fused.append(self._merge_span_group(current_span_group, text))

        return fused

    def _merge_span_group(self, group: list[DetectorResult], text: str) -> FusedSpan:
        """Merge a group of overlapping spans into a single FusedSpan."""
        # For simplicity, we'll take the type and text from the span with the highest confidence.
        # A more advanced strategy could use a type hierarchy.
        best_span = max(group, key=lambda s: s.confidence)
        start = min(s.start for s in group)
        end = max(s.end for s in group)
        sources = tuple(sorted({s.detector_name for s in group}))
        max_confidence = max(s.confidence for s in group)

        return FusedSpan(
            pii_type=best_span.pii_type,
            text=text[start:end],
            start=start,
            end=end,
            sources=sources,
            max_confidence=max_confidence,
        )
