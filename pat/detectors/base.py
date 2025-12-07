"""Base classes for detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class DetectorResult:
    """A raw detection from a single detector."""

    pii_type: str
    text: str
    start: int
    end: int
    confidence: float
    detector_name: str
    validator_passed: bool | None = None


@dataclass(frozen=True)
class DetectorContext:
    """Contextual information provided to detectors."""

    prior_results: Sequence[DetectorResult] = field(default_factory=tuple)


class BaseDetector(ABC):
    """Abstract base class for all PII detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the detector."""
        ...

    @abstractmethod
    def run(self, text: str, context: DetectorContext | None = None) -> list[DetectorResult]:
        """Execute the detector on the input text."""
        ...