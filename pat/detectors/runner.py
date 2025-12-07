"""
The DetectorRunner orchestrates the execution of multiple PII detectors.

It is responsible for loading and running only the detectors that are
explicitly enabled in the application configuration.
"""

from __future__ import annotations

import logging
from typing import Callable

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

        # Map of config keys → detector factories.
        # Some keys are aliases for the same underlying detector type.
        detector_factories: dict[str, Callable[[], BaseDetector]] = {
            "regex": lambda: RegexDetector(),
            "ner": lambda: NERDetector(alias_name="ner"),
            "ml_ner": lambda: NERDetector(alias_name="ml_ner"),
            "embedding": lambda: EmbeddingSimilarityDetector(alias_name="embedding"),
            "semantic": lambda: EmbeddingSimilarityDetector(alias_name="semantic"),
            "domain": lambda: DomainHeuristicsDetector(),
            "domain_heuristic": lambda: DomainHeuristicsDetector(),
            "ml_token": lambda: MLTokenClassifierDetector(),
        }

        # Canonicalization: these config names share the same underlying detector.
        canonical_key = {
            "semantic": "embedding",
            "domain": "domain_heuristic",
        }

        added_names: set[str] = set()
        added_canonical: set[str] = set()

        for key, factory in detector_factories.items():
            # Skip if disabled in settings (default False if missing).
            if not settings.detector_enabled.get(key, False):
                continue

            canon = canonical_key.get(key, key)
            if canon in added_canonical:
                continue

            try:
                detector = factory()
            except Exception as exc:  # pragma: no cover - defensive
                LOG.error("Failed to instantiate detector '%s': %s", key, exc)
                continue

            # Avoid duplicated detector names in case of alias collisions.
            if detector.name in added_names:
                continue

            self.detectors.append(detector)
            added_names.add(detector.name)
            added_canonical.add(canon)
            LOG.info("Detector enabled: %s (config key: %s)", detector.name, key)

        if not self.detectors:
            LOG.warning("DetectorRunner initialised with no active detectors.")

    def run(self, text: str) -> list[DetectorResult]:
        """Run all loaded detectors on the input text in sequence.

        The order of detectors in `self.detectors` defines the order of execution.
        Each detector receives a DetectorContext with all prior results so far.
        """
        if not text:
            return []

        results: list[DetectorResult] = []

        for detector in self.detectors:
            context = DetectorContext(prior_results=tuple(results))
            try:
                # Use the BaseDetector.execute wrapper to enforce invariants.
                detector_results = detector.execute(text, context=context)
            except Exception as exc:  # pragma: no cover - defensive
                LOG.error(
                    "Detector '%s' raised an exception during execution: %s",
                    detector.name,
                    exc,
                )
                continue

            if detector_results:
                results.extend(detector_results)

        return results
