"""Heuristic, context-aware PII detector."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Set, Tuple, Sequence

from pat.config.heuristics import HEURISTIC_RULES, HEURISTIC_VERSION
from pat.detectors.base import BaseDetector, DetectorContext, DetectorResult
from pat.utils.text import trim_span

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CompiledHeuristicRule:
    """Internal normalized representation of a heuristic rule."""
    id: str
    pii_type: str
    any_keywords: tuple[str, ...]
    all_keywords: tuple[str, ...]
    context_window: int
    base_score: float
    requires_digits: bool
    number_pattern: re.Pattern[str] | None
    right_context: int
    priority: int
    keyword_regex: re.Pattern[str]


class DomainHeuristicsDetector(BaseDetector):
    """Detector that uses contextual keywords to find PII.

    Design goals:
        - Acts as a precision booster, not a primary firehose.
        - Keeps spans short, keyword-scoped, and bounded.
        - Deduplicates overlapping spans across rules/windows.
        - Tolerates partial/mismatched HEURISTIC_RULES schema gracefully.
    """

    # Global bounds to prevent runaway masking
    MIN_SPAN_LEN = 3
    MAX_SPAN_LEN = 64

    def __init__(self) -> None:
        compiled_rules: List[_CompiledHeuristicRule] = []

        for raw in HEURISTIC_RULES:
            try:
                rule_id = getattr(raw, "id", "")
                pii_type = getattr(raw, "pii_type", None)
                if not pii_type:
                    LOG.warning("Heuristic rule '%s' missing pii_type; skipping.", rule_id)
                    continue

                any_keywords = tuple(getattr(raw, "any_keywords", ()) or ())
                if not any_keywords:
                    # By design: rules without keywords are considered too broad.
                    LOG.debug(
                        "Heuristic rule '%s' has no any_keywords; skipping by design.",
                        rule_id,
                    )
                    continue

                all_keywords = tuple(getattr(raw, "all_keywords", ()) or ())
                context_window = int(getattr(raw, "context_window", 48) or 48)
                base_score = float(getattr(raw, "base_score", 0.75) or 0.75)
                requires_digits = bool(getattr(raw, "requires_digits", False))
                number_pattern = getattr(raw, "number_pattern", None)
                right_context = int(getattr(raw, "right_context", 24) or 24)
                priority = int(getattr(raw, "priority", 100) or 100)

                # Normalise number_pattern to compiled regex if provided as string.
                compiled_num_pattern: re.Pattern[str] | None = None
                if number_pattern is not None:
                    if isinstance(number_pattern, re.Pattern):
                        compiled_num_pattern = number_pattern
                    elif isinstance(number_pattern, str):
                        compiled_num_pattern = re.compile(number_pattern)
                    else:
                        LOG.warning(
                            "Heuristic rule '%s' has unsupported number_pattern type %r; ignoring.",
                            rule_id,
                            type(number_pattern),
                        )

                # Build keyword regex once per rule.
                escaped_keywords = [re.escape(kw) for kw in any_keywords]
                keyword_regex = re.compile(
                    r"\b(" + "|".join(escaped_keywords) + r")\b",
                    re.IGNORECASE,
                )

                compiled_rules.append(
                    _CompiledHeuristicRule(
                        id=rule_id,
                        pii_type=pii_type,
                        any_keywords=any_keywords,
                        all_keywords=all_keywords,
                        context_window=max(8, context_window),
                        base_score=max(0.0, min(1.0, base_score)),
                        requires_digits=requires_digits,
                        number_pattern=compiled_num_pattern,
                        right_context=max(0, right_context),
                        priority=priority,
                        keyword_regex=keyword_regex,
                    )
                )
            except Exception:
                LOG.exception("Failed to normalize heuristic rule %r; skipping.", raw)
                continue

        # Deterministic ordering: lower priority first, then by id.
        self._rules: Sequence[_CompiledHeuristicRule] = sorted(
            compiled_rules, key=lambda r: (r.priority, r.id)
        )

        LOG.info(
            "DomainHeuristicsDetector initialised with %d rules (version=%s).",
            len(self._rules),
            HEURISTIC_VERSION,
        )

    @property
    def name(self) -> str:
        return "domain_heuristic"

    def run(self, text: str, context: DetectorContext | None = None) -> list[DetectorResult]:
        """Run all configured heuristic rules against the text."""
        if not text:
            return []

        results: List[DetectorResult] = []
        text_lower = text.lower()
        text_len = len(text)

        # Deduplicate across rules/windows: key = (start, end, pii_type)
        seen_spans: Set[Tuple[int, int, str]] = set()

        for rule in self._rules:
            # Pre-filter: require at least one 'any_keyword' in the full text
            if not any(kw.lower() in text_lower for kw in rule.any_keywords):
                continue

            # Optional: all_keywords must also be present in the full text
            if rule.all_keywords and not all(
                kw.lower() in text_lower for kw in rule.all_keywords
            ):
                continue

            # If digits required but we have no usable numeric pattern, skip rule.
            if rule.requires_digits and rule.number_pattern is None:
                continue

            # Slide over each keyword occurrence
            for kw_match in rule.keyword_regex.finditer(text):
                # Context window around the keyword
                window_start = max(0, kw_match.start() - rule.context_window)
                window_end = min(text_len, kw_match.end() + rule.context_window)
                window_text = text[window_start:window_end]

                if rule.requires_digits:
                    self._collect_numeric_spans(
                        text=text,
                        window_text=window_text,
                        window_start=window_start,
                        rule=rule,
                        seen_spans=seen_spans,
                        results=results,
                    )
                else:
                    self._collect_keyword_span(
                        text=text,
                        kw_match=kw_match,
                        rule=rule,
                        text_len=text_len,
                        seen_spans=seen_spans,
                        results=results,
                    )

        return results

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _collect_numeric_spans(
        self,
        *,
        text: str,
        window_text: str,
        window_start: int,
        rule: _CompiledHeuristicRule,
        seen_spans: Set[Tuple[int, int, str]],
        results: List[DetectorResult],
    ) -> None:
        """Collect numeric spans within a keyword-scoped window."""
        assert rule.number_pattern is not None

        for num_match in rule.number_pattern.finditer(window_text):
            span_start = window_start + num_match.start()
            span_end = window_start + num_match.end()
            span_text = num_match.group(0)

            if len(span_text) < self.MIN_SPAN_LEN or len(span_text) > self.MAX_SPAN_LEN:
                continue

            key = (span_start, span_end, rule.pii_type)
            if key in seen_spans:
                continue
            seen_spans.add(key)

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

    def _collect_keyword_span(
        self,
        *,
        text: str,
        kw_match: re.Match,
        rule: _CompiledHeuristicRule,
        text_len: int,
        seen_spans: Set[Tuple[int, int, str]],
        results: List[DetectorResult],
    ) -> None:
        """Collect non-numeric spans anchored around the keyword."""
        kw_start = kw_match.start()
        kw_end = kw_match.end()

        raw_end = min(text_len, kw_end + rule.right_context)
        span_start, span_end, span_text = trim_span(text, kw_start, raw_end)

        if len(span_text) < self.MIN_SPAN_LEN or len(span_text) > self.MAX_SPAN_LEN:
            return

        key = (span_start, span_end, rule.pii_type)
        if key in seen_spans:
            return
        seen_spans.add(key)

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
