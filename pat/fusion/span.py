"""Dataclasses for the fusion layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from pat.detectors.base import DetectorResult
import logging

LOG = logging.getLogger(__name__)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


@dataclass
class FusedSpan:
    """
    Represents the merged evidence across detectors for a PII span.
    The FusionEngine guarantees non-overlapping, correctly ordered spans.
    """

    start: int
    end: int
    text: str
    pii_type: str

    confidence: float = 0.0
    category: str | None = None

    all_types: set[str] = field(default_factory=set)
    all_categories: set[str] = field(default_factory=set)
    detectors: set[str] = field(default_factory=set)
    detector_scores: dict[str, float] = field(default_factory=dict)
    sources: list[DetectorResult] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    left_context: str = ""
    right_context: str = ""

    # Severity fields (populated post-classifier)
    max_confidence: float | None = None
    severity_score: float | None = None
    severity_label: str | None = None
    severity_probs: dict[str, float] | None = None

    # ---------------------------------------
    # Construction helpers
    # ---------------------------------------
    @classmethod
    def from_detector_result(cls, result: DetectorResult) -> FusedSpan:
        """Create a FusedSpan from a single detector result."""
        score = getattr(result, "score", None)
        conf = _safe_float(score if score is not None else result.confidence, default=0.0)

        metadata = getattr(result, "metadata", {}) or {}
        pattern_id = metadata.get("pattern_id")

        return cls(
            start=result.start,
            end=result.end,
            text=result.text,
            pii_type=result.pii_type,
            category=getattr(result, "category", None),
            confidence=conf,
            all_types={result.pii_type},
            all_categories={c for c in [getattr(result, "category", None)] if c},
            detectors={result.detector_name},
            detector_scores={result.detector_name: conf},
            sources=[result],
            max_confidence=conf,
            metadata={
                "pattern_id": pattern_id,
                "detector_metadata": [metadata] if metadata else []
            },
        )

    # ---------------------------------------
    # Dataclass invariant enforcement
    # ---------------------------------------
    def __post_init__(self) -> None:
        # Enforce start < end
        if self.end < self.start:
            LOG.warning(
                f"Invalid span boundaries: start={self.start}, end={self.end}. Fixing to non-negative length."
            )
            self.end = max(self.start, self.end)

        # Ensure required sets
        self.all_types.add(self.pii_type)

        if self.category:
            self.all_categories.add(self.category)

        # Infer detector list from sources
        if not self.detectors and self.sources:
            names = [getattr(src, "detector_name", None) for src in self.sources]
            self.detectors = {n for n in names if n}

        # Infer category from taxonomy
        if self.category is None and self.pii_type:
            try:
                from pat.utils.taxonomy import category_for_type

                inferred = category_for_type(self.pii_type)
                if inferred:
                    self.category = inferred
                    self.all_categories.add(inferred)
            except Exception as e:
                LOG.error(f"Failed to infer category for type {self.pii_type}: {e}")

        # Ensure confidence fields are sane
        self.confidence = _safe_float(self.confidence, 0.0)
        if self.max_confidence is None:
            self.max_confidence = self.confidence

        # Ensure context fields are strings
        self.left_context = self.left_context or ""
        self.right_context = self.right_context or ""

    # ---------------------------------------
    # Properties
    # ---------------------------------------
    @property
    def start_char(self) -> int:
        return self.start

    @property
    def end_char(self) -> int:
        return self.end

    @property
    def primary_type(self) -> str:
        return self.pii_type

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)

    # ---------------------------------------
    # Merge additional detector evidence
    # ---------------------------------------
    def add_source(self, result: DetectorResult) -> None:
        """Append detector evidence and refresh bookkeeping."""

        self.sources.append(result)
        self.all_types.add(result.pii_type)

        # Category merge
        cat = getattr(result, "category", None)
        if cat:
            self.all_categories.add(cat)

        # Detector name
        name = result.detector_name
        if name:
            self.detectors.add(name)

        # Score merge
        score = getattr(result, "score", None)
        incoming_conf = _safe_float(score if score is not None else result.confidence)

        self.detector_scores[name] = incoming_conf
        self.max_confidence = max(self.max_confidence or 0.0, incoming_conf)

        # Metadata merge
        metadata = getattr(result, "metadata", None)
        if metadata:
            self.metadata.setdefault("detector_metadata", []).append(metadata)
