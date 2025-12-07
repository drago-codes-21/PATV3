"""Regex-based PII detector."""

from __future__ import annotations

import logging

from pat.config.patterns import get_compiled_patterns
from pat.detectors.base import BaseDetector, DetectorContext, DetectorResult
from pat.validators import VALIDATORS

LOG = logging.getLogger(__name__)


class RegexDetector(BaseDetector):
    """Detector that uses a library of configured regex patterns."""

    def __init__(self) -> None:
        """Initialise the detector."""
        self.patterns = get_compiled_patterns()

    @property
    def name(self) -> str:
        """Return the unique name of the detector."""
        return "regex"

    def run(self, text: str, context: DetectorContext | None = None) -> list[DetectorResult]:
        """Run all configured regex patterns against the text."""
        results: list[DetectorResult] = []
        lowered = text.lower()
        for pattern in self.patterns:
            if pattern.keywords and not any(kw.lower() in lowered for kw in pattern.keywords):
                continue
            for match in pattern.regex.finditer(text):
                validator_passed = None
                if pattern.validator:
                    validator_func = VALIDATORS.get(pattern.validator)
                    if not validator_func:
                        LOG.warning("Validator '%s' not found.", pattern.validator)
                        continue
                    if not validator_func(match):
                        continue
                    validator_passed = True
                # Skip financial-looking matches explicitly marked as references/invoices/tickets or product IDs.
                if pattern.pii_type in {"BANK_ACCOUNT", "SORT_CODE", "CARD_NUMBER", "CUSTOMER_ID", "IBAN"}:
                    window_start = max(0, match.start() - 24)
                    window_end = min(len(text), match.end() + 24)
                    window = text[window_start:window_end].lower()
                    if any(token in window for token in ("reference", "ticket", "invoice", "inv-", "product id", "sku")):
                        continue

                results.append(
                    DetectorResult(
                        pii_type=pattern.pii_type,
                        text=match.group(pattern.group or 0),
                        start=match.start(pattern.group or 0),
                        end=match.end(pattern.group or 0),
                        confidence=pattern.confidence,
                        detector_name=self.name,
                        validator_passed=validator_passed,
                    )
                )
        return results
