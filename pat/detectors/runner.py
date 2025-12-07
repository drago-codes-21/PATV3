"""
The DetectorRunner orchestrates the execution of multiple PII detectors.

It is responsible for loading and running only the detectors that are
explicitly enabled in the application configuration.
"""

from __future__ import annotations

import logging
from typing import Sequence

from pat.config import get_settings

from .base import BaseDetector, DetectorContext, DetectorResult
from .domain_heuristics_detector import DomainHeuristicsDetector
from .embedding_detector import EmbeddingSimilarityDetector
from .ml_token_classifier import MLTokenClassifierDetector
from .ner_detector import NERDetector
from .regex_detector import RegexDetector

LOG = logging.getLogger(__name__)


class DetectorRunner:
    """Loads and runs all enabled detectors."""

    def __init__(self) -> None:
        """Initialise the runner by loading detectors from config."""
        settings = get_settings()
        self.detectors: list[BaseDetector] = []

        # A map of all possible detectors available in the system.
        all_detectors: dict[str, type[BaseDetector]] = {
            "regex": RegexDetector,
            "ner": NERDetector,
            "ml_ner": NERDetector,
            "embedding": EmbeddingSimilarityDetector,
            "semantic": EmbeddingSimilarityDetector,
            "domain": DomainHeuristicsDetector,
            "domain_heuristic": DomainHeuristicsDetector,
            "ml_token": MLTokenClassifierDetector,
        }

        added_names: set[str] = set()
        # Dynamically load only the detectors enabled in the settings.
        for name, detector_class in all_detectors.items():
            if not settings.detector_enabled.get(name, False):
                continue
            detector = detector_class()
            if detector.name in added_names:
                continue
            self.detectors.append(detector)
            added_names.add(detector.name)
            LOG.info("Detector enabled: %s", detector.name)

    def run(self, text: str) -> list[DetectorResult]:
        """Run all loaded detectors on the input text."""
        results: list[DetectorResult] = []
        for detector in self.detectors:
            context = DetectorContext(prior_results=tuple(results))
            results.extend(detector.run(text, context=context))
        return results
