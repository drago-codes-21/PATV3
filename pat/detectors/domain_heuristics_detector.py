"""Heuristic, context-aware PII detector."""

from __future__ import annotations

import re

from pat.config.heuristics import HEURISTIC_RULES
from pat.detectors.base import BaseDetector, DetectorContext, DetectorResult
from pat.utils.text import trim_span


class DomainHeuristicsDetector(BaseDetector):
    """Detector that uses contextual keywords to find PII."""

    @property
    def name(self) -> str:
        """Return the unique name of the detector."""
        return "domain_heuristic"

    def run(self, text: str, context: DetectorContext | None = None) -> list[DetectorResult]:
        """Run all configured heuristic rules against the text."""
        results: list[DetectorResult] = []
        text_lower = text.lower()

        for rule in HEURISTIC_RULES:
            # Simple keyword check first for performance
            if not any(kw in text_lower for kw in rule.any_keywords):
                continue

            keyword_regex = re.compile(r"\b(" + "|".join(rule.any_keywords) + r")\b", re.IGNORECASE)

            for kw_match in keyword_regex.finditer(text):
                # Define a window around the keyword
                start = max(0, kw_match.start() - rule.context_window)
                end = min(len(text), kw_match.end() + rule.context_window)
                window_text = text[start:end]

                if rule.requires_digits:
                    for num_match in rule.number_pattern.finditer(window_text):
                        results.append(
                            DetectorResult(
                                pii_type=rule.pii_type,
                                text=num_match.group(0),
                                start=start + num_match.start(),
                                end=start + num_match.end(),
                                confidence=rule.base_score,
                                detector_name=self.name,
                            )
                        )
                else:
                    # For non-numeric rules, keep the span tight around the keyword to avoid over-masking.
                    kw_start = kw_match.start()
                    kw_end = kw_match.end()
                    span_start, span_end, span_text = trim_span(text, kw_start, min(len(text), kw_end + 16))
                    if len(span_text) < 3:
                        continue
                    results.append(
                        DetectorResult(
                            pii_type=rule.pii_type,
                            text=span_text,
                            start=span_start,
                            end=span_end,
                            confidence=rule.base_score,
                            detector_name=self.name,
                        )
                    )
        return results
