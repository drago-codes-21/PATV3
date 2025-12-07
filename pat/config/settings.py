"""
Global settings and configuration helpers for PAT.

This module loads all runtime configuration for detectors, fusion,
severity scoring, embeddings, and policy rules. It validates paths,
normalizes thresholds, and provides a frozen Settings object to ensure
deterministic, reproducible behaviour across the pipeline.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional

# ------------------------------------------------------------
# Settings Dataclass
# ------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    """Centralised, immutable configuration object."""

    project_root: Path

    # Model paths
    embedding_model_path: Path
    severity_model_path: Path

    # Config files
    policy_file_path: Path
    detector_thresholds_path: Path

    # Detector configuration
    detector_thresholds: Dict[str, Dict[str, float]]
    detector_weights: Dict[str, float]
    detector_enabled: Dict[str, bool]

    # NER / ML models
    ner_model_name_or_path: str
    ner_confidence: float

    # Severity configuration
    severity_thresholds: Dict[str, Tuple[float, float]]
    severity_type_thresholds: Dict[str, Dict[str, float | Tuple[float, float]]]
    severity_probability_thresholds: Dict[str, float]

    # Embeddings configuration
    embedding_similarity_threshold: float
    embedding_sentence_split_chars: Tuple[str, ...] = ("\n", ".", "!", "?")

    # Debug/logging flags
    debug_detectors: bool = False
    debug_fusion: bool = False

    # Fusion behaviour
    fusion_context_window: int = 80

    def severity_label(self, score: float) -> str:
        """
        Convert a severity score in [0, 1] to a severity label.
        Falls back to CRITICAL if thresholds are exhausted.
        """
        if not 0.0 <= score <= 1.0:
            return "CRITICAL"

        for label, (low, high) in self.severity_thresholds.items():
            if low <= score < high:
                return label

        return "CRITICAL"


# ------------------------------------------------------------
# Global settings cache
# ------------------------------------------------------------

_SETTINGS: Optional[Settings] = None


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _env_path(key: str, default: Path) -> Path:
    """Resolve a path from environment or fall back to default."""
    value = os.getenv(key)
    try:
        return Path(value).expanduser() if value else default
    except Exception:
        return default


def _load_json(path: Path, *, default: Any) -> Any:
    """Safe JSON loader with clear error logs."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# ------------------------------------------------------------
# Load Settings
# ------------------------------------------------------------

def _load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]

    # Paths
    artifacts_dir = project_root / "artifacts"
    embedding_dir = project_root / "pat" / "embeddings" / "model"
    severity_models_dir = project_root / "pat" / "models" / "severity"

    # Allow overrides via environment
    embedding_model_path = _env_path("PAT_EMBEDDING_MODEL_PATH", embedding_dir)

    # Severity model with pointer handling
    severity_model_path = _env_path(
        "PAT_SEVERITY_MODEL_PATH",
        artifacts_dir / "span_severity_model.joblib",
    )

    pointer_file = severity_models_dir / "latest.txt"
    if pointer_file.exists():
        try:
            model_path_in_pointer = pointer_file.read_text().strip()
            # Resolve path relative to project root for robustness.
            resolved = project_root / model_path_in_pointer
            if resolved.is_file():
                severity_model_path = resolved
        except Exception:  # pragma: no cover
            pass  # Keep default if pointer is invalid

    # Detector thresholds
    detector_thresholds_path = _env_path(
        "PAT_DETECTOR_THRESHOLDS_PATH",
        project_root / "pat" / "config" / "detector_thresholds.json",
    )
    detector_thresholds = _load_json(detector_thresholds_path, default={})

    # Policy rules file (JSON)
    policy_file_path = _env_path(
        "PAT_POLICY_FILE_PATH",
        project_root / "pat" / "config" / "policy_rules.json",
    )

    # Detector weights — cleaned, only active detectors
    detector_weights = {
        "regex": 1.0,
        "ner": 0.9,
        "ml_ner": 0.9,
        "embedding": 0.7,
        "domain_heuristic": 0.8,
        "ml_token": 0.85,
    }

    detector_enabled = {k: True for k in detector_weights.keys()}

    # Severity thresholds (updated to match new policy config)
    severity_thresholds = {
        "LOW": (0.0, 0.25),
        "MEDIUM": (0.25, 0.60),
        "HIGH": (0.60, 0.85),
        "VERY_HIGH": (0.85, 1.01),
    }

    # Type-specific overrides (empty for now; populated by CIP)
    severity_type_thresholds: Dict[str, Dict[str, float | Tuple[float, float]]] = {}

    # Probability cutoff for highest-confidence class selection
    severity_probability_thresholds = {
        "VERY_HIGH": 0.60,
        "HIGH": 0.45,
        "MEDIUM": 0.20,
        "LOW": 0.0,
    }

    # Embedding similarity gate
    embedding_similarity_threshold = float(
        os.getenv("PAT_EMBEDDING_SIMILARITY_THRESHOLD", "0.55")
    )

    ner_model_name_or_path = os.getenv("PAT_NER_MODEL_PATH", "en_core_web_sm")
    ner_confidence = float(os.getenv("PAT_NER_CONFIDENCE", "0.75"))

    debug_detectors = os.getenv("PAT_DEBUG_DETECTORS", "").lower() in {"1", "true", "yes"}
    debug_fusion = os.getenv("PAT_DEBUG_FUSION", "").lower() in {"1", "true", "yes"}
    fusion_context_window = int(os.getenv("PAT_FUSION_CONTEXT_WINDOW", "80"))

    return Settings(
        project_root=project_root,
        embedding_model_path=embedding_model_path,
        severity_model_path=severity_model_path,
        policy_file_path=policy_file_path,
        detector_thresholds_path=detector_thresholds_path,
        detector_thresholds=detector_thresholds,
        detector_weights=detector_weights,
        detector_enabled=detector_enabled,
        ner_model_name_or_path=ner_model_name_or_path,
        ner_confidence=ner_confidence,
        severity_thresholds=severity_thresholds,
        severity_type_thresholds=severity_type_thresholds,
        severity_probability_thresholds=severity_probability_thresholds,
        embedding_similarity_threshold=embedding_similarity_threshold,
        debug_detectors=debug_detectors,
        debug_fusion=debug_fusion,
        fusion_context_window=fusion_context_window,
    )


# ------------------------------------------------------------
# Public Settings Getter
# ------------------------------------------------------------

def get_settings() -> Settings:
    """Return a cached Settings instance."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = _load_settings()
    return _SETTINGS
