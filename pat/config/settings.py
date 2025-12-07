"""Global settings and configuration helpers for PAT."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Settings:
    """Centralised configuration object."""

    project_root: Path
    embedding_model_path: Path
    severity_model_path: Path
    policy_file_path: Path
    detector_thresholds_path: Path
    detector_thresholds: Dict[str, Dict[str, float]]
    ner_model_name_or_path: str
    detector_weights: Dict[str, float]
    detector_enabled: Dict[str, bool]
    severity_thresholds: Dict[str, Tuple[float, float]]
    severity_type_thresholds: Dict[str, Dict[str, float | Tuple[float, float]]]
    severity_probability_thresholds: Dict[str, float]
    ner_confidence: float
    embedding_similarity_threshold: float = 0.55
    embedding_sentence_split_chars: Tuple[str, ...] = ("\n", ".", "!", "?")
    debug_detectors: bool = False
    debug_fusion: bool = False
    fusion_context_window: int = 80

    def severity_label(self, score: float) -> str:
        """Map a score in [0, 1] to a severity label."""

        for label, (low, high) in self.severity_thresholds.items():
            if low <= score < high:
                return label
        return "CRITICAL"


_SETTINGS: Settings | None = None


def _env_path(key: str, default: Path) -> Path:
    value = os.getenv(key)
    return Path(value).expanduser() if value else default


def _load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    artifacts_dir = project_root / "artifacts"
    embedding_dir = project_root / "pat" / "embeddings" / "model"
    severity_models_dir = project_root / "pat" / "models" / "severity"

    embedding_model_path = _env_path("PAT_EMBEDDING_MODEL_PATH", embedding_dir)
    severity_model_path = _env_path(
        "PAT_SEVERITY_MODEL_PATH", artifacts_dir / "span_severity_model.joblib"
    )
    pointer_file = severity_models_dir / "latest.txt"
    if pointer_file.exists():
        try:
            pointer_val = Path(pointer_file.read_text(encoding="utf-8").strip())
            if pointer_val.exists():
                severity_model_path = pointer_val
        except Exception:
            pass
    detector_thresholds_path = _env_path(
        "PAT_DETECTOR_THRESHOLDS_PATH", project_root / "pat" / "config" / "detector_thresholds.json"
    )
    policy_file_path = _env_path(
        "PAT_POLICY_FILE_PATH", project_root / "pat" / "config" / "policy_rules.json"
    )
    ner_model_name_or_path = os.getenv(
        "PAT_NER_MODEL_PATH", "en_core_web_sm"
    )
    ner_confidence = float(os.getenv("PAT_NER_CONFIDENCE", "0.75"))
    debug_detectors = os.getenv("PAT_DEBUG_DETECTORS", "false").lower() in {"1", "true", "yes"}
    debug_fusion = os.getenv("PAT_DEBUG_FUSION", "false").lower() in {"1", "true", "yes"}
    fusion_context_window = int(os.getenv("PAT_FUSION_CONTEXT_WINDOW", "80"))

    detector_weights = {
        "regex": 1.0,
        "ner": 0.9,
        "ml_ner": 0.9,
        "embedding": 0.6,
        "semantic": 0.6,
        "domain": 0.8,
        "domain_heuristic": 0.8,
        "ml_token": 0.85,
    }
    detector_enabled = {name: True for name in detector_weights}

    severity_thresholds = {
        "LOW": (0.0, 0.2),
        "MEDIUM": (0.2, 0.6),
        "HIGH": (0.6, 0.85),
        "VERY_HIGH": (0.85, 1.01),
    }
    severity_type_thresholds: Dict[str, Dict[str, float | Tuple[float, float]]] = {}
    severity_probability_thresholds: Dict[str, float] = {
        "VERY_HIGH": 0.6,
        "HIGH": 0.45,
        "MEDIUM": 0.2,
        "LOW": 0.0,
    }

    embedding_similarity_threshold = float(os.getenv("PAT_EMBEDDING_SIMILARITY_THRESHOLD", "0.55"))
    detector_thresholds: Dict[str, Dict[str, float]] = {}
    if detector_thresholds_path.exists():
        try:
            detector_thresholds = json.loads(detector_thresholds_path.read_text(encoding="utf-8"))
        except Exception:
            detector_thresholds = {}

    return Settings(
        project_root=project_root,
        embedding_model_path=embedding_model_path,
        severity_model_path=severity_model_path,
        policy_file_path=policy_file_path,
        detector_thresholds_path=detector_thresholds_path,
        detector_thresholds=detector_thresholds,
        ner_model_name_or_path=ner_model_name_or_path,
        detector_weights=detector_weights,
        detector_enabled=detector_enabled,
        severity_thresholds=severity_thresholds,
        severity_type_thresholds=severity_type_thresholds,
        severity_probability_thresholds=severity_probability_thresholds,
        ner_confidence=ner_confidence,
        embedding_similarity_threshold=embedding_similarity_threshold,
        debug_detectors=debug_detectors,
        debug_fusion=debug_fusion,
        fusion_context_window=fusion_context_window,
    )

def get_settings() -> Settings:
    """Return cached settings."""

    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = _load_settings()
    return _SETTINGS
