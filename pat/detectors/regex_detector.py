"""
Production-grade Regex-based PII detector.

Major improvements:
    - strict validator safety
    - consistent group resolution (supports named groups)
    - safer keyword gating
    - reduced financial false positives
    - max-length sanity limit to stop runaway spans
    - clean confidence coercion
    - warning logs for disabled or filtered patterns
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from pat.config.patterns import (
    ALLOWED_LITERALS,
    NEGATIVE_PATTERN_IDS,
    PATTERN_VERSION,
    PatternDefinition,
    get_compiled_patterns,
)
from pat.detectors.base import BaseDetector, DetectorContext, DetectorResult
from pat.validators import VALIDATORS

LOG = logging.getLogger(__name__)


class RegexDetector(BaseDetector):

    MAX_SPAN_CHARS = 128   # safety bound for runaway matches
    KEYWORD_WINDOW = 48    # smaller = safer for PRECISION
    FINANCIAL_NEG_WINDOW = 32

    def __init__(self) -> None:
        raw_patterns = get_compiled_patterns()

        # Disable patterns via config
        filtered_patterns = []
        for p in raw_patterns:
            if p.id in NEGATIVE_PATTERN_IDS:
                LOG.warning("RegexDetector: pattern '%s' disabled via NEGATIVE_PATTERN_IDS.", p.id)
                continue
            filtered_patterns.append(p)

        validated_patterns: List[PatternDefinition] = []

        # validator integrity check
        for p in filtered_patterns:
            if p.validator and p.validator not in VALIDATORS:
                LOG.error(
                    "RegexDetector: Pattern '%s' references unknown validator '%s'; skipping.",
                    p.id,
                    p.validator,
                )
                continue
            validated_patterns.append(p)

        # deterministic ordering: priority asc → ID asc
        validated_patterns.sort(key=lambda pat: (pat.priority, pat.id))

        self.patterns = validated_patterns

        LOG.info(
            "RegexDetector initialised with %d patterns (version=%s).",
            len(self.patterns),
            PATTERN_VERSION,
        )

    @property
    def name(self) -> str:
        return "regex"

    # ----------------------------------------------------------------------
    # Core detection logic
    # ----------------------------------------------------------------------

    def run(self, text: str, context: DetectorContext | None = None) -> List[DetectorResult]:
        if not text:
            return []

        results: List[DetectorResult] = []
        lowered = text.lower()
        text_len = len(text)

        for pattern in self.patterns:
            compiled = pattern.regex
            if isinstance(compiled, str):
                compiled = re.compile(compiled, pattern.flags)

            for match in compiled.finditer(text):
                span_text, start, end = self._extract_group(pattern, match)
                if span_text is None:
                    continue

                # guard: runaway spans
                if (end - start) > self.MAX_SPAN_CHARS:
                    LOG.warning(
                        "RegexDetector: pattern '%s' produced oversized span (%d chars). Skipped.",
                        pattern.id,
                        end - start,
                    )
                    continue

                # skip allowlisted fragments (contains OR exact)
                lowered_span = span_text.lower()
                if span_text in ALLOWED_LITERALS or any(lit.lower() in lowered_span for lit in ALLOWED_LITERALS):
                    continue

                # keyword gating
                if pattern.keywords:
                    if not self._keyword_gate(pattern, lowered, start, end):
                        continue

                # validator enforcement
                if pattern.validator:
                    validator = VALIDATORS.get(pattern.validator)
                    try:
                        if not validator(match):
                            continue
                    except Exception:
                        LOG.exception(
                            "RegexDetector: validator '%s' for pattern '%s' raised an exception.",
                            pattern.validator,
                            pattern.id,
                        )
                        continue

                # reduce financial FPs in business text
                if pattern.pii_type in {"CARD_NUMBER", "SORT_CODE", "BANK_ACCOUNT", "CUSTOMER_ID", "IBAN"}:
                    if self._looks_like_business_reference(pattern, lowered, start, end):
                        continue

                results.append(
                    DetectorResult(
                        pii_type=pattern.pii_type,
                        text=span_text,
                        start=start,
                        end=end,
                        confidence=float(pattern.confidence or 1.0),
                        detector_name=self.name,
                        metadata={"pattern_id": pattern.id},
                    )
                )

        return results

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _extract_group(self, pattern: PatternDefinition, match: re.Match) -> Tuple[str | None, int, int]:
        """Safe extraction of group or named group."""
        try:
            group = pattern.group
            if group is None:
                span_text = match.group(0)
                return span_text, match.start(0), match.end(0)

            # support named groups
            if isinstance(group, str):
                span_text = match.group(group)
                return span_text, match.start(group), match.end(group)

            # numeric groups
            span_text = match.group(group)
            return span_text, match.start(group), match.end(group)

        except Exception:
            LOG.warning(
                "RegexDetector: pattern '%s' group=%r is invalid for this match; skipping.",
                pattern.id,
                pattern.group,
            )
            return None, -1, -1

    def _keyword_gate(self, pattern: PatternDefinition, lowered_text: str, start: int, end: int) -> bool:
        """Require at least one keyword in a small window around match."""
        window_start = max(0, start - self.KEYWORD_WINDOW)
        window_end = min(len(lowered_text), end + self.KEYWORD_WINDOW)
        window = lowered_text[window_start:window_end]

        for kw in pattern.keywords:
            if kw.lower() in window:
                return True
        return False

    def _looks_like_business_reference(self, pattern: PatternDefinition, lowered: str, start: int, end: int) -> bool:
        """Prevent masking invoice references / SKUs labelled as bank IDs."""
        neg_window = lowered[max(0, start - self.FINANCIAL_NEG_WINDOW): min(len(lowered), end + self.FINANCIAL_NEG_WINDOW)]

        business_terms = (
            "invoice",
            "inv-",
            "reference",
            "ref ",
            "ticket",
            "sku",
            "product id",
            "order id",
        )

        # Must contain *multiple* business tokens to treat as reference
        signal_count = sum(1 for t in business_terms if t in neg_window)
        if signal_count >= 2:
            # drop low-confidence financial regex hits
            if float(pattern.confidence or 0) < 0.75:
                return True

        # Explicit guard for product identifiers that resemble card numbers.
        if pattern.pii_type == "CARD_NUMBER" and ("product id" in neg_window or "product" in neg_window):
            return True

        return False
