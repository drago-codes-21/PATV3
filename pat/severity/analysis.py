"""Severity feature analysis, importance, and drift monitoring."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pat.severity.features import FEATURE_NAMES, assert_feature_schema, span_features_to_vector

LOG = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import joblib
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("joblib and pandas are required for pat.severity.analysis") from exc


def _load_model(path: Path):
    model = joblib.load(path)
    names = getattr(model, "feature_names_in_", FEATURE_NAMES)
    assert_feature_schema(names)
    return model


def _load_jsonl_features(path: Path) -> np.ndarray:
    rows: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            feats = payload.get("features") or payload.get("feature_vector") or payload
            if isinstance(feats, dict):
                vector = span_features_to_vector(feats)
            else:
                arr = np.asarray(feats, dtype=float)
                if arr.shape[0] != len(FEATURE_NAMES):
                    raise AssertionError("Feature vector length mismatch in input JSONL.")
                vector = arr
            rows.append(vector)
    if not rows:
        raise ValueError(f"No feature rows found in {path}")
    return np.vstack(rows)


def _compute_stats(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    stats = {}
    for idx, name in enumerate(FEATURE_NAMES):
        stats[name] = {
            "mean": float(means[idx]),
            "std": float(stds[idx]),
            "min": float(mins[idx]),
            "max": float(maxs[idx]),
        }
    return stats


def _drift(train_stats: dict[str, Any], infer_stats: dict[str, Any]) -> dict[str, float]:
    drift_scores: dict[str, float] = {}
    for name in FEATURE_NAMES:
        t_mean = float(train_stats[name]["mean"])
        i_mean = float(infer_stats[name]["mean"])
        t_std = float(train_stats[name].get("std", 0.0)) or 1e-6
        drift_scores[name] = abs(i_mean - t_mean) / t_std
    return drift_scores


def _feature_importance(model) -> dict[str, float]:
    importances: Iterable[float]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "booster_"):
        importances = model.booster_.feature_importance()
    else:
        importances = [0.0 for _ in FEATURE_NAMES]
    return {name: float(val) for name, val in zip(FEATURE_NAMES, importances)}


def _render_markdown(
    model_path: Path,
    importance: dict[str, float],
    train_stats: dict[str, Any] | None,
    infer_stats: dict[str, Any],
    drift_scores: dict[str, float] | None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Severity Analysis Report\n")
    lines.append(f"- Model: `{model_path}`")
    lines.append(f"- Features: {len(FEATURE_NAMES)} (schema locked)\n")
    lines.append("## Feature Importance (top 10)\n")
    top = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:10]
    for name, val in top:
        lines.append(f"- {name}: {val:.4f}")
    lines.append("\n## Summary Statistics (inference)")
    for name in FEATURE_NAMES:
        stats = infer_stats[name]
        lines.append(
            f"- {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"min={stats['min']:.4f}, max={stats['max']:.4f}"
        )
    if train_stats and drift_scores:
        lines.append("\n## Drift Scores (|mean_delta|/train_std, top 10)")
        for name, score in sorted(drift_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            lines.append(f"- {name}: {score:.3f}")
    return "\n".join(lines)


def run_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL of span features.")
    parser.add_argument("--model", type=Path, required=True, help="Severity model path.")
    parser.add_argument("--train-stats", type=Path, help="Optional training_stats.json")
    parser.add_argument("--output", type=Path, help="Markdown/HTML output path.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    model = _load_model(args.model)
    features = _load_jsonl_features(args.input)
    infer_stats = _compute_stats(features)
    importance = _feature_importance(model)

    train_stats = None
    drift_scores = None
    if args.train_stats and args.train_stats.exists():
        with args.train_stats.open("r", encoding="utf-8") as handle:
            train_stats = json.load(handle)
        drift_scores = _drift(train_stats, infer_stats)

    report = _render_markdown(args.model, importance, train_stats, infer_stats, drift_scores)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        LOG.info("Report written to %s", args.output)
    else:
        print(report)


if __name__ == "__main__":  # pragma: no cover
    run_cli()
