"""Severity model inference component."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from pat.config import get_settings

LOG = logging.getLogger(__name__)


class SeverityModel:
    """Loads and runs the trained severity model for inference."""

    def __init__(self, model_path: Path | None = None) -> None:
        """Initialise by loading the model from disk."""
        settings = get_settings()
        self.path = model_path or settings.severity_model_path
        self.model = None

        if self.path.exists():
            try:
                self.model = joblib.load(self.path)
                LOG.info("Severity model loaded from %s", self.path)
            except Exception:
                LOG.exception("Failed to load severity model from %s", self.path)
        else:
            LOG.warning("Severity model not found at %s. Severity scoring will be disabled.", self.path)

    def predict(self, features: list) -> tuple[float, str, dict[str, float]]:
        """Predict severity score, label, and probabilities for a given feature vector."""
        if not self.model:
            return 0.0, "LOW", {"LOW": 1.0}

        # The model expects a 2D array
        probabilities = self.model.predict_proba([features])[0]
        class_labels = self.model.classes_

        probs_dict = {label: prob for label, prob in zip(class_labels, probabilities)}

        # A common way to create a single score is a weighted average.
        # Let's define weights corresponding to the severity levels.
        label_to_weight = {"LOW": 0.1, "MEDIUM": 0.5, "HIGH": 0.8, "VERY_HIGH": 1.0}

        # Get weights in the order of model classes
        weights = np.array([label_to_weight.get(label, 0.0) for label in class_labels])

        score = float(np.dot(probabilities, weights))

        # The label is the one with the highest probability
        predicted_label = class_labels[np.argmax(probabilities)]

        return score, predicted_label, probs_dict