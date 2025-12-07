"""PAT Continuous Improvement Pipeline (CIP) orchestrator."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from pat.calibration import calibrate_thresholds as calib
from pat.config import get_settings
from pat.detectors.embedding_detector import EmbeddingSimilarityDetector
from pat.detectors.ml_token_classifier import MLTokenClassifierDetector
from pat.severity.features import assert_feature_schema, FEATURE_NAMES
from pat.severity.analysis import _compute_stats, _drift, _load_jsonl_features  # type: ignore
from pat.cip import report as cip_report

LOG = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_cmd(cmd: List[str], env: dict[str, str] | None = None) -> None:
    LOG.info("Running command: %s", " ".join(cmd))
    res = subprocess.run(cmd, env=env or os.environ.copy())
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _compute_f1(scores: List[tuple[float, bool]], threshold: float) -> float:
    tp = sum(1 for s, ok in scores if s >= threshold and ok)
    fp = sum(1 for s, ok in scores if s >= threshold and not ok)
    fn = sum(1 for s, ok in scores if s < threshold and ok)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _collect_scores(detector, rows: list[dict]) -> dict[str, list[tuple[float, bool]]]:
    per_type: dict[str, list[tuple[float, bool]]] = {}
    for row in rows:
        text = row.get("text") or ""
        gold = row.get("spans") or []
        detections = detector.run(text)
        for det in detections:
            match = calib._match_detection(det, gold)
            per_type.setdefault(det.pii_type, []).append((det.confidence, match))
    return per_type


def _run_regression_tests() -> str:
    try:
        import pytest  # noqa: F401
        _run_cmd(
            ["python", "-m", "pytest", "tests/test_severity_feature_schema.py", "tests/detector_layer_validation.py", "tests/run_manual_validations.py"]
        )
        return "passed"
    except ImportError:
        LOG.warning("pytest not installed; falling back to manual validation scripts.")
        _run_cmd(["python", "tests/run_manual_validations.py"])
        return "manual"


def _write_thresholds(thresholds: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")


def _adopt_thresholds(rows: list[dict], artifact_dir: Path, settings) -> dict[str, Any]:
    results = calib.calibrate(rows)
    current_embedding = EmbeddingSimilarityDetector()
    current_ml = MLTokenClassifierDetector()

    accepted_embedding: dict[str, float] = {}
    rejected: list[str] = []

    # Evaluate baseline F1 per type
    emb_scores = _collect_scores(current_embedding, rows)
    for pii_type, scores in emb_scores.items():
        baseline = _compute_f1(scores, current_embedding.type_thresholds.get(pii_type, current_embedding.threshold))
        best = results["embedding"].get(pii_type, {})
        best_f1 = best.get("f1", 0.0)
        if best_f1 >= baseline * 0.95:
            accepted_embedding[pii_type] = best.get("threshold", current_embedding.type_thresholds.get(pii_type, current_embedding.threshold))
        else:
            rejected.append(pii_type)

    accepted_ml: dict[str, float] = {}
    ml_scores = _collect_scores(current_ml, rows)
    for pii_type, scores in ml_scores.items():
        baseline = _compute_f1(scores, current_ml.thresholds.get(pii_type, current_ml.DEFAULT_THRESHOLD))
        best = results["ml_token"].get(pii_type, {})
        best_f1 = best.get("f1", 0.0)
        if best_f1 >= baseline * 0.95:
            accepted_ml[pii_type] = best.get("threshold", current_ml.thresholds.get(pii_type, current_ml.DEFAULT_THRESHOLD))
        else:
            rejected.append(pii_type)

    payload = {"embedding": accepted_embedding, "ml_token": accepted_ml}
    ts = _timestamp()
    snapshot = artifact_dir / f"{ts}_thresholds.json"
    _write_thresholds(payload, snapshot)
    # Update live thresholds file
    _write_thresholds(payload, settings.detector_thresholds_path)
    LOG.info("Thresholds adopted: %s", payload)
    return {"payload": payload, "rejected": rejected, "snapshot": snapshot}


def _maybe_enable_debug(debug_flag: bool, corpus: Path, limit: int, debug_dir: Path) -> None:
    if not debug_flag or not corpus.exists():
        return
    LOG.warning("DEBUG ENABLED: Unredacted PII will be written to %s", debug_dir / "debug_subset.jsonl")
    debug_lines: list[str] = []
    with corpus.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            debug_lines.append(line)
    debug_subset = debug_dir / "debug_subset.jsonl"
    _ensure_dir(debug_dir)
    debug_subset.write_text("".join(debug_lines), encoding="utf-8")
    env = os.environ.copy()
    env["PAT_DEBUG_DETECTORS"] = "true"
    try:
        _run_cmd(
            [
                "python",
                "-m",
                "pat.evaluation.run_e2e_report",
                "--input",
                str(debug_subset),
                "--output-json",
                str(debug_dir / "debug_report.json"),
                "--output-md",
                str(debug_dir / "debug_report.md"),
            ],
            env=env,
        )
    finally:
        env.pop("PAT_DEBUG_DETECTORS", None)


def _run_e2e(corpus: Path, json_path: Path, md_path: Path, model_path: Path | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    if model_path:
        env["PAT_SEVERITY_MODEL_PATH"] = str(model_path)
    _run_cmd(
        [
            "python",
            "-m",
            "pat.evaluation.run_e2e_report",
            "--input",
            str(corpus),
            "--output-json",
            str(json_path),
            "--output-md",
            str(md_path),
        ],
        env=env,
    )
    return json.loads(json_path.read_text(encoding="utf-8"))


def _maybe_retrain_severity(config: dict[str, Any], artifact_dir: Path, baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    train_data = config.get("training_data")
    drift_features = config.get("drift_features")
    train_stats_path = config.get("drift_train_stats")
    drift_threshold = float(config.get("drift_threshold", 0.15))
    if not drift_features or not train_stats_path:
        return {"status": "skipped"}
    infer_matrix = _load_jsonl_features(Path(drift_features))
    infer_stats = _compute_stats(infer_matrix)
    train_stats = json.loads(Path(train_stats_path).read_text(encoding="utf-8"))
    drift_scores = _drift(train_stats, infer_stats)
    max_drift = max(drift_scores.values()) if drift_scores else 0.0
    if max_drift < drift_threshold:
        return {"status": "no_drift", "max_drift": max_drift}

    LOG.info("Drift detected (%.3f >= %.3f), retraining severity model.", max_drift, drift_threshold)
    ts = _timestamp()
    model_dir = Path("pat") / "models" / "severity"
    _ensure_dir(model_dir)
    candidate_model = model_dir / f"{ts}_severity_model.joblib"
    from pat.severity import train as severity_train

    severity_train.train(
        [
            "--input",
            train_data or "data/span_training.csv",
            "--output",
            str(candidate_model),
        ]
    )
    pointer = model_dir / "latest.txt"
    model_old = pointer.read_text(encoding="utf-8").strip() if pointer.exists() else ""
    pointer.write_text(str(candidate_model), encoding="utf-8")

    # Validate candidate
    corpus = Path(config.get("e2e_corpus", "data/e2e.jsonl"))
    cand_json = artifact_dir / f"{ts}_e2e_candidate.json"
    cand_md = artifact_dir / f"{ts}_e2e_candidate.md"
    candidate_metrics = _run_e2e(corpus, cand_json, cand_md, model_path=candidate_model)
    if baseline_metrics and candidate_metrics.get("f1", 0.0) < baseline_metrics.get("f1", 0.0) * 0.98:
        # rollback
        LOG.warning("Candidate model regressed F1; rolling back to %s", model_old)
        if model_old:
            pointer.write_text(model_old, encoding="utf-8")
        return {
            "status": "rolled_back",
            "max_drift": max_drift,
            "candidate_model": str(candidate_model),
            "baseline_model": model_old,
            "candidate_metrics": candidate_metrics,
        }
    return {
        "status": "promoted",
        "max_drift": max_drift,
        "candidate_model": str(candidate_model),
        "candidate_metrics": candidate_metrics,
    }


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="YAML config for CIP.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    config = _load_yaml_config(args.config)
    settings = get_settings()

    artifact_root = Path(config.get("artifacts_dir", "pat/cip/artifacts"))
    artifact_dir = _ensure_dir(artifact_root / _timestamp())
    debug_dir = artifact_dir / "debug_runs"

    # 1. Regression tests
    regression_status = _run_regression_tests()

    # 2. Optional threshold calibration
    calibration_applied = False
    calibration_rejected: list[str] = []
    thresholds_payload = {}
    calib_data = config.get("calibration_data")
    if calib_data and config.get("enable_calibration", True):
        rows = calib._load_jsonl(Path(calib_data))
        calib_result = _adopt_thresholds(rows, artifact_dir, settings)
        calibration_applied = True
        calibration_rejected = calib_result["rejected"]
        thresholds_payload = calib_result["payload"]

    # 3. Severity feature schema check
    assert_feature_schema(FEATURE_NAMES)

    # 4. Drift + retraining
    e2e_corpus = Path(config.get("e2e_corpus", "data/e2e.jsonl"))
    baseline_json = artifact_dir / "e2e_baseline.json"
    baseline_md = artifact_dir / "e2e_baseline.md"
    baseline_metrics = _run_e2e(e2e_corpus, baseline_json, baseline_md, model_path=None)
    retrain_info = _maybe_retrain_severity(config, artifact_dir, baseline_metrics)

    # 5. Debug logging on subset
    _maybe_enable_debug(config.get("enable_debug_subset", False), e2e_corpus, int(config.get("debug_subset_limit", 20)), debug_dir)

    # 6. Final report
    payload: Dict[str, Any] = {
        "regression_status": regression_status,
        "regression_details": "",
        "calibration_applied": calibration_applied,
        "calibration_rejected": calibration_rejected,
        "thresholds_old": settings.detector_thresholds,
        "thresholds_new": thresholds_payload,
        "drift_max": retrain_info.get("max_drift"),
        "drift_threshold": config.get("drift_threshold", 0.15),
        "retrain_status": retrain_info.get("status"),
        "model_path": retrain_info.get("candidate_model") or str(settings.severity_model_path),
        "model_old": retrain_info.get("baseline_model"),
        "model_new": retrain_info.get("candidate_model"),
        "e2e_baseline": baseline_metrics,
        "e2e_candidate": retrain_info.get("candidate_metrics"),
        "multi_detector_spans": baseline_metrics.get("multi_detector_spans"),
        "single_detector_spans": baseline_metrics.get("single_detector_spans"),
        "suggested_actions": [],
    }
    report_md = artifact_dir / f"{_timestamp()}_cip_report.md"
    report_json = artifact_dir / f"{_timestamp()}_cip_report.json"
    cip_report.write_report(payload, report_md, report_json)
    LOG.info("CIP completed. Reports: %s, %s", report_md, report_json)


if __name__ == "__main__":  # pragma: no cover
    run()
