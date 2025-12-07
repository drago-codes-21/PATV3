"""Sweep detector thresholds to maximise F1 on labeled spans."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from pat.detectors.embedding_detector import EmbeddingSimilarityDetector
from pat.detectors.ml_token_classifier import MLTokenClassifierDetector
from pat.config import get_settings

LOG = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def _match_detection(det, gold_spans: list[dict]) -> bool:
    best_iou = 0.0
    for gold in gold_spans:
        if gold.get("pii_type") != det.pii_type:
            continue
        best_iou = max(best_iou, _iou(det.start, det.end, gold["start"], gold["end"]))
    return best_iou >= 0.5


def _collect_scores(detector, rows: list[dict]) -> dict[str, list[tuple[float, bool]]]:
    per_type: dict[str, list[tuple[float, bool]]] = {}
    for row in rows:
        text = row.get("text") or ""
        gold = row.get("spans") or []
        detections = detector.run(text)
        for det in detections:
            success = _match_detection(det, gold)
            per_type.setdefault(det.pii_type, []).append((det.confidence, success))
    return per_type


def _sweep(scores: list[tuple[float, bool]], thresholds: Iterable[float]) -> tuple[float, float]:
    best_threshold = 0.0
    best_f1 = -1.0
    for thresh in thresholds:
        tp = sum(1 for score, ok in scores if score >= thresh and ok)
        fp = sum(1 for score, ok in scores if score >= thresh and not ok)
        fn = sum(1 for score, ok in scores if score < thresh and ok)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    return best_threshold, best_f1


def calibrate(rows: list[dict], num_steps: int = 17) -> dict:
    thresholds = np.linspace(0.2, 0.9, num_steps)

    embedding = EmbeddingSimilarityDetector()
    embedding.threshold = 0.0
    embedding.type_thresholds = {k: 0.0 for k in embedding.type_thresholds}

    ml = MLTokenClassifierDetector()
    ml.thresholds = {k: 0.0 for k in ml.thresholds}

    results: dict[str, dict] = {"embedding": {}, "ml_token": {}}
    emb_scores = _collect_scores(embedding, rows)
    ml_scores = _collect_scores(ml, rows)

    for pii_type, scores in emb_scores.items():
        best_t, best_f1 = _sweep(scores, thresholds)
        results["embedding"][pii_type] = {"threshold": best_t, "f1": best_f1}

    for pii_type, scores in ml_scores.items():
        best_t, best_f1 = _sweep(scores, thresholds)
        results["ml_token"][pii_type] = {"threshold": best_t, "f1": best_f1}

    return results


def run_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="JSONL with text and spans.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write thresholds JSON.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="If set, apply thresholds to the configured detector thresholds path.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    rows = _load_jsonl(args.data)
    LOG.info("Loaded %d labeled rows", len(rows))
    results = calibrate(rows)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOG.info("Wrote calibrated thresholds to %s", args.output)

    if args.apply:
        settings = get_settings()
        payload = {"embedding": {}, "ml_token": {}}
        for pii_type, vals in results.get("embedding", {}).items():
            payload["embedding"][pii_type] = vals.get("threshold")
        for pii_type, vals in results.get("ml_token", {}).items():
            payload["ml_token"][pii_type] = vals.get("threshold")
        settings.detector_thresholds_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOG.info("Applied thresholds to %s", settings.detector_thresholds_path)


if __name__ == "__main__":  # pragma: no cover
    run_cli()
