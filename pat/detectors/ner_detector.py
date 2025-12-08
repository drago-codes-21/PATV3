"""spaCy NER-based detector (production-hardened)."""

from __future__ import annotations

import logging
from typing import List, Optional, Set, Tuple

from pat.config import get_settings
from pat.detectors.base import BaseDetector, DetectorContext, DetectorResult
from pat.utils.text import trim_span

LOG = logging.getLogger(__name__)

# spaCy import is optional; detector degrades gracefully.
try:  # pragma: no cover - optional dependency
    import spacy
    from spacy.language import Language
except ImportError:  # pragma: no cover - optional dependency
    spacy = None
    Language = None


# -------------------------------------------------------------------
# PII taxonomy mapping
# -------------------------------------------------------------------
# Intentionally conservative:
#   - Only high-signal NER labels are mapped.
#   - Company/org names (ORG) are *not* automatically treated as PII.
#
SPACY_LABEL_MAP: dict[str, Optional[str]] = {
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE",
    "MONEY": "MONEY",
    # Explicitly ignored
    "ORG": None,
    "PRODUCT": None,
    "EVENT": None,
    "WORK_OF_ART": None,
    "LAW": None,
    "LANGUAGE": None,
    "TIME": None,
    "PERCENT": None,
    "QUANTITY": None,
    "ORDINAL": None,
    "CARDINAL": None,
}


class NERDetector(BaseDetector):
    """spaCy NER detector with strict safety rules to avoid over-masking."""

    MIN_SPAN_LEN = 2
    MAX_SPAN_LEN = 80

    def __init__(self, alias_name: str | None = None) -> None:
        self._name = alias_name or "ner"

        if spacy is None:
            LOG.warning("spaCy not installed; NER detector '%s' will be disabled.", self._name)
            self.nlp = None
            self.model_name = None
            self.confidence = 0.0
            return

        settings = get_settings()
        self.model_name = settings.ner_model_name_or_path
        # Clamp confidence into [0, 1]
        self.confidence = max(0.0, min(1.0, float(getattr(settings, "ner_confidence", 0.75))))

        try:
            self.nlp: Optional[Language] = spacy.load(self.model_name)  # type: ignore[arg-type]
            LOG.info(
                "spaCy NER model '%s' loaded for detector '%s'.",
                self.model_name,
                self._name,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOG.error(
                "Failed to load spaCy NER model '%s': %s. Detector '%s' will be disabled.",
                self.model_name,
                exc,
                self._name,
            )
            self.nlp = None

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Main detection logic
    # ------------------------------------------------------------------
    def run(self, text: str, context: DetectorContext | None = None) -> List[DetectorResult]:
        """Run the spaCy model and extract mapped PII entities."""
        if not text or not self.nlp:
            return []

        try:
            doc = self.nlp(text)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.error("spaCy NER inference failed in '%s': %s", self._name, exc)
            return []

        results: List[DetectorResult] = []
        seen: Set[Tuple[int, int, str]] = set()
        text_len = len(text)

        for ent in getattr(doc, "ents", ()):
            label = getattr(ent, "label_", None)
            if not label:
                continue

            mapped_type = SPACY_LABEL_MAP.get(label)
            if mapped_type is None:
                continue  # non-PII entity or explicitly ignored

            start_char = getattr(ent, "start_char", None)
            end_char = getattr(ent, "end_char", None)
            if start_char is None or end_char is None:
                continue

            # Guard against obviously broken offsets from spaCy/custom pipes.
            if not (0 <= start_char < end_char <= text_len):
                LOG.debug(
                    "Skipping NER span with invalid offsets (%s, %s) for label=%s.",
                    start_char,
                    end_char,
                    label,
                )
                continue

            # Trim whitespace / punctuation at edges while preserving offsets.
            adj_start, adj_end, adj_text = trim_span(text, start_char, end_char)

            if not adj_text:
                continue

            if adj_text.lower() in {"git", "repo", "github", "bitbucket"}:
                continue

            # Skip all-caps headings to reduce over-masking of section titles.
            if adj_text.isupper() and len(adj_text.split()) <= 3:
                continue

            # Drop numeric-only spaCy DATE guesses to avoid masking benign counters/IDs.
            if mapped_type == "DATE" and adj_text.isdigit():
                continue

            span_len = len(adj_text)
            if span_len < self.MIN_SPAN_LEN or span_len > self.MAX_SPAN_LEN:
                continue

            # Ignore purely whitespace or punctuation spans.
            if adj_text.strip() == "":
                continue

            key = (adj_start, adj_end, mapped_type)
            if key in seen:
                continue
            seen.add(key)

            results.append(
                DetectorResult(
                    pii_type=mapped_type,
                    text=adj_text,
                    start=adj_start,
                    end=adj_end,
                    confidence=self.confidence,
                    detector_name=self.name,
                )
            )

        return results
