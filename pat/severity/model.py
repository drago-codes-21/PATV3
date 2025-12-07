"""Severity model inference component."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import json

from pat.config import get_settings
from pat.severity.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

LOG = logging.getLogger(__name__)


class SeverityModel:
    """Loads and runs the trained severity model for inference."""

    EXPECTED_CLASSES = ("LOW", "MEDIUM", "HIGH", "VERY_HIGH")

    def __init__(self, model_path: Path | None = None) -> None:
        """Initialise by loading the model from disk."""
        settings = get_settings()
        self.path = model_path or settings.severity_model_path
        self.model = None

        if self.path.exists():
            try:
                self.model = joblib.load(self.path)
                self._validate_model(self.model)
                LOG.info("Severity model loaded from %s", self.path)
            except Exception as exc:
                LOG.exception("Failed to load severity model from %s", self.path)
                raise
        else:
            LOG.warning("Severity model not found at %s. Severity scoring will be disabled.", self.path)

    def predict(self, features: list) -> tuple[float, str, dict[str, float]]:
        """Predict severity score, label, and probabilities for a given feature vector."""
        if not self.model:
            return 0.0, "LOW", {"LOW": 1.0}

        # The model expects a 2D array
        probabilities = self.model.predict_proba([features])[0]
        class_labels = list(getattr(self.model, "classes_", []))

        probs_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities)}

        # Severity score is a weighted average using explicit label weights.
        label_to_weight = {"LOW": 0.1, "MEDIUM": 0.5, "HIGH": 0.8, "VERY_HIGH": 1.0}
        try:
            weights = np.array([label_to_weight[label] for label in class_labels], dtype=float)
        except KeyError as exc:
            raise ValueError(f"Unexpected severity class in model: {exc}") from exc

        score = float(np.dot(probabilities, weights))

        predicted_label = class_labels[int(np.argmax(probabilities))]

        return score, predicted_label, probs_dict

    def _validate_model(self, model: object) -> None:
        """Ensure loaded model aligns with expected feature schema and labels."""

        meta_path = Path(self.path).with_suffix(".metadata.json")
        metadata = None
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                LOG.warning("Could not read severity metadata from %s", meta_path)

        classes_attr = getattr(model, "classes_", None)
        classes = list(classes_attr) if classes_attr is not None else None
        if classes is None and metadata:
            classes = metadata.get("class_labels")
        if classes is None:
            raise ValueError("Severity model is missing required attribute 'classes_'.")
        if set(classes) != set(self.EXPECTED_CLASSES):
            raise ValueError(
                f"Severity model classes mismatch. Expected {self.EXPECTED_CLASSES}, got {tuple(classes)}"
            )

        feature_names_attr = getattr(model, "feature_names_in_", None)
        if feature_names_attr is None:
            feature_names_attr = getattr(model, "pat_feature_names", None)
        if feature_names_attr is None and metadata:
            feature_names_attr = metadata.get("feature_names")
        feature_names = list(feature_names_attr) if feature_names_attr is not None else None
        if feature_names is None:
            raise ValueError(
                "Severity model missing feature names metadata; retrain with current FEATURE_NAMES."
            )
        if list(feature_names) != FEATURE_NAMES:
            raise ValueError(
                "Severity model feature schema mismatch "
                f"(expected version {FEATURE_SCHEMA_VERSION})."
            )

        schema_version = getattr(model, "schema_version", None) or (metadata and metadata.get("schema_version"))
        if not schema_version:
            raise ValueError(
                "Severity model missing schema_version metadata; retrain to align with FEATURE_SCHEMA."
            )
        if schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Severity model schema_version mismatch. Expected {FEATURE_SCHEMA_VERSION}, got {schema_version}"
            )
