"""
Base classes and core types for PII detectors.

This layer is deliberately defensive:
    - Enforces span invariants early (fail-fast behaviour)
    - Normalises DetectorResult fields
    - Provides debug hooks without polluting detector logic
    - Guarantees consistency for FusionEngine and Severity layers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence, List


# ---------------------------------------------------------------------------
# Utility validation helpers (internal)
# ---------------------------------------------------------------------------

def _validate_offsets(start: int, end: int, text: str) -> None:
    """Ensure `0 <= start < end <= len(text)`."""
    if start < 0 or end < 0:
        raise ValueError(f"DetectorResult offsets cannot be negative: ({start}, {end})")
    if start >= end:
        raise ValueError(f"DetectorResult must satisfy start < end: ({start}, {end})")
    if end > len(text):
        raise ValueError(
            f"DetectorResult end index {end} out of bounds for text length {len(text)}"
        )


def _validate_span_text(span_text: str, original: str, start: int, end: int) -> None:
    """Ensure result.text == original[start:end]."""
    expected = original[start:end]
    if span_text != expected:
        # This is dangerous; detectors MUST return the exact substring.
        raise ValueError(
            f"DetectorResult.text mismatch: expected='{expected}', got='{span_text}' "
            f"for span=({start},{end})"
        )


# ---------------------------------------------------------------------------
# Detector output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectorResult:
    """A raw detection from a single detector.

    Invariants:
        - 0 <= start < end <= len(text)
        - `text` must equal the substring of the input document at (start, end)
        - pii_type is a canonical taxonomy label
        - confidence ∈ [0.0, 1.0]
    """

    pii_type: str
    text: str
    start: int
    end: int
    confidence: float
    detector_name: str
    validator_passed: bool | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def span(self) -> tuple[int, int]:
        return self.start, self.end

    @property
    def length(self) -> int:
        return self.end - self.start


# ---------------------------------------------------------------------------
# Optional context for detectors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectorContext:
    """Context passed into detectors.

    Contains:
        - prior_results: outputs from detectors executed earlier in the pipeline.
    """

    prior_results: Sequence[DetectorResult] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Detector base class (production-grade contract)
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """
    Abstract base class for all PII detectors.

    Produces DetectorResult objects that MUST be:
        - Immutable
        - Offset-accurate
        - Span-text consistent
        - Deterministic (same input → same outputs)

    Detectors *must not* mutate the input text.
    """

    # ------------------------------------------------------------------
    # Required properties and behaviour
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Return unique detector name used throughout the pipeline."""
        ...

    @abstractmethod
    def run(self, text: str, context: DetectorContext | None = None) -> list[DetectorResult]:
        """
        Execute the detector.

        Implementors must:
            - Return spans relative to *exactly* this `text`
            - Ensure no silent failures, no negative offsets, no swapped spans
            - Keep detectors stateless

        Returns:
            List of DetectorResult
        """
        ...

    # ------------------------------------------------------------------
    # Optional: safe post-processing hook for implementors
    # ------------------------------------------------------------------

    def _postprocess_results(self, text: str, results: List[DetectorResult]) -> List[DetectorResult]:
        """
        Validate DetectorResult invariants and return a safe, clean list.

        This function:
            - Enforces span boundaries
            - Confirms substring correctness
            - Ensures deterministic ordering of outputs (start asc, end asc)
            - Deduplicates identical spans from buggy detectors

        All detectors automatically receive this behaviour unless overridden.
        """

        sane: List[DetectorResult] = []
        seen: set[tuple[str, int, int]] = set()

        for res in results:
            try:
                _validate_offsets(res.start, res.end, text)
                _validate_span_text(res.text, text, res.start, res.end)
            except Exception as exc:
                # Fail-fast but without killing the entire pipeline.
                # Log + skip; fusion must not ingest corrupted spans.
                import logging
                logging.getLogger(__name__).error(
                    "Invalid DetectorResult from %s: %s", res.detector_name, exc
                )
                continue

            key = (res.pii_type, res.start, res.end)
            if key in seen:
                continue

            seen.add(key)
            sane.append(res)

        sane.sort(key=lambda r: (r.start, r.end))
        return sane

    # ------------------------------------------------------------------
    # Public wrapper execution used by pipeline
    # ------------------------------------------------------------------

    def execute(self, text: str, context: DetectorContext | None = None) -> list[DetectorResult]:
        """
        Safe entry point for pipeline to run detectors.

        Applies:
            - defensive input checks
            - post-processing
            - validation of every result
        """

        if not isinstance(text, str):
            raise TypeError(f"Detector input must be a string, got {type(text)}")

        raw_results = self.run(text, context=context)
        if not raw_results:
            return []

        # Validate + clean
        return self._postprocess_results(text, raw_results)
