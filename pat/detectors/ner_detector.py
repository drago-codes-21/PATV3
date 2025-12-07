"""spaCy NER-based detector."""

from __future__ import annotations

import logging

from pat.config import get_settings
from pat.detectors.base import BaseDetector, DetectorContext, DetectorResult

LOG = logging.getLogger(__name__)

try:
    import spacy
    from spacy.language import Language
except ImportError:
    spacy = None
    Language = None


# A simple mapping from spaCy's default NER labels to the PAT taxonomy.
# This should be expanded for more comprehensive models.
SPACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",  # Geopolitical Entity
    "LOC": "LOCATION",
    "DATE": "DATE",
    "MONEY": "MONEY",
    "PRODUCT": None,  # Often noisy, map to None to ignore
    "EVENT": None,
    "WORK_OF_ART": None,
    "LAW": None,
    "LANGUAGE": None,
    "TIME": None,
    "PERCENT": None,
    "QUANTITY": None,
    "ORDINAL": "GENERIC_NUMBER",
    "CARDINAL": "GENERIC_NUMBER",
}


class NERDetector(BaseDetector):
    """Detector that uses a pre-trained spaCy NER model."""

    def __init__(self) -> None:
        """Initialise the detector by loading the spaCy model."""
        if spacy is None:
            raise ImportError("spaCy is not installed. Please run: pip install spacy")

        settings = get_settings()
        self.model_name = settings.ner_model_name_or_path
        self.confidence = settings.ner_confidence
        self.nlp: Language | None = None
        try:
            self.nlp = spacy.load(self.model_name)
            LOG.info("spaCy NER model '%s' loaded successfully.", self.model_name)
        except OSError:
            LOG.error(
                "Could not load spaCy model '%s'. "
                "Please run 'python -m spacy download %s' or provide a valid path.",
                self.model_name,
                self.model_name,
            )
            # Allow initialization to succeed but the run method will be a no-op.
            self.nlp = None

    @property
    def name(self) -> str:
        """Return the unique name of the detector."""
        return "ner"

    def run(self, text: str, context: DetectorContext | None = None) -> list[DetectorResult]:
        """Run the spaCy NER model over the text."""
        if not self.nlp:
            return []

        doc = self.nlp(text)
        results: list[DetectorResult] = []
        for ent in doc.ents:
            pii_type = SPACY_LABEL_MAP.get(ent.label_)

            # Ignore entities that are not mapped to a PII type.
            if pii_type is None:
                continue

            ent_text = getattr(ent, "text", None) or text[ent.start_char : ent.end_char]

            results.append(
                DetectorResult(
                    pii_type=pii_type,
                    text=ent_text,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=self.confidence,
                    detector_name=self.name,
                )
            )
        return results
