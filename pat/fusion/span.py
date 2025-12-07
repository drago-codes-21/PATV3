"""Dataclasses for the fusion layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from pat.detectors.base import DetectorResult


@dataclass
class FusedSpan:
    """Result of merging detector hits."""

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
    max_confidence: float | None = None
    severity_score: float | None = None
    severity_label: str | None = None
    severity_probs: dict[str, float] | None = None

    @classmethod
    def from_detector_result(cls, result: DetectorResult) -> FusedSpan:
        """Create a FusedSpan from a single detector result."""

        score_val = getattr(result, "score", None)
        return cls(
            start=result.start,
            end=result.end,
            text=result.text,
            pii_type=result.pii_type,
            category=getattr(result, "category", None),
            confidence=float(score_val if score_val is not None else result.confidence),
            all_types={result.pii_type},
            all_categories={c for c in [getattr(result, "category", None)] if c},
            detectors={result.detector_name},
            detector_scores={result.detector_name: float(score_val if score_val is not None else result.confidence)},
            max_confidence=float(score_val if score_val is not None else result.confidence),
            sources=[result],
            metadata={
                "pattern_id": getattr(result, "metadata", None) and result.metadata.get("pattern_id"),
                "detector_metadata": [getattr(result, "metadata", {})] if getattr(result, "metadata", None) else [],
            },
        )

    def __post_init__(self) -> None:
        if not self.all_types:
            self.all_types = {self.pii_type}
        if not self.all_categories and self.category:
            self.all_categories = {self.category}
        if not self.detectors and self.sources:
            try:
                self.detectors = {src.detector_name for src in self.sources if hasattr(src, "detector_name")}
                if not self.detectors:
                    self.detectors = {str(src) for src in self.sources}
            except Exception:
                self.detectors = set()
        if self.category is None and self.pii_type:
            try:
                from pat.utils.taxonomy import category_for_type

                self.category = category_for_type(self.pii_type)
                if self.category:
                    self.all_categories.add(self.category)
            except Exception:
                ...
        if self.max_confidence is None:
            self.max_confidence = self.confidence

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

    def add_source(self, result: DetectorResult) -> None:
        """Append detector evidence and refresh bookkeeping."""

        self.sources.append(result)
        self.all_types.add(result.pii_type)
        cat = getattr(result, "category", None)
        if cat:
            self.all_categories.add(cat)
        self.detectors.add(result.detector_name)
        score_val = getattr(result, "score", None)
        self.detector_scores[result.detector_name] = float(
            score_val if score_val is not None else result.confidence
        )
        metadata = getattr(result, "metadata", None)
        if metadata:
            self.metadata.setdefault("detector_metadata", []).append(metadata)
        self.max_confidence = max(self.max_confidence or 0.0, self.confidence)
