"""Training entrypoint for span-level severity classification."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pat.config import get_settings
from pat.detectors.runner import DetectorRunner
from pat.embeddings import EmbeddingModel
from pat.fusion import FusedSpan, FusionEngine
from pat.severity.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    assert_feature_schema,
    compute_neighbor_stats,
    compute_span_sentence_context,
    compute_token_position_ratio,
    get_span_context_text,
    extract_span_features,
    span_features_to_vector,
)
from pat.severity.model import SeverityModel
from pat.utils.text import compute_sentence_boundaries

LOG = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import joblib
except ImportError as exc:  # pragma: no cover
    raise SystemExit("joblib is required for training") from exc

try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split, StratifiedKFold
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required for training") from exc

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pandas is required for training data de-duplication."
    ) from exc


CLASS_LIST = ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
LABEL_MAP = {
    "CRITICAL": "VERY_HIGH",
    "VERY HIGH": "VERY_HIGH",
    "VERY_HIGH": "VERY_HIGH",
    "HIGH": "HIGH",
    "MEDIUM": "MEDIUM",
    "LOW": "LOW",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/span_training.csv"),
        help="CSV or JSONL containing span-level severity annotations.",
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--span-text-column", default="span_text")
    parser.add_argument("--span-start-column", default="span_start")
    parser.add_argument("--span-end-column", default="span_end")
    parser.add_argument("--pii-type-column", default="pii_type")
    parser.add_argument("--label-column", default="severity_label")
    parser.add_argument(
        "--output",
        type=Path,
        default=get_settings().severity_model_path,
        help="Where to save the trained span severity model.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("pat/models/severity/reports"),
        help="Directory to write training reports (stats, importance).",
    )
    parser.add_argument(
        "--disable-detector-augmentation",
        action="store_true",
        help="Skip running detectors to enrich spans with source metadata.",
    )
    parser.add_argument(
        "--validation-split-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset to use for validation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for the train/validation split.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of cross-validation folds for stability checks.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for LightGBM.",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=31,
        help="Number of leaves for LightGBM.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=-1,
        help="Max depth for LightGBM trees.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=150,
        help="Number of estimators for LightGBM.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["none", "sigmoid", "isotonic"],
        default="sigmoid",
        help="Apply probability calibration using the given method.",
    )
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


class SpanDatasetBuilder:
    """Utility for constructing span feature matrices."""

    def __init__(self, use_detectors: bool = True) -> None:
        self.use_detectors = use_detectors
        self.runner = DetectorRunner() if use_detectors else None
        self.fusion = FusionEngine() if use_detectors else None
        self.embedding_model = EmbeddingModel()
        # Instantiate a separate runner for data repair to avoid caching issues.
        self.repair_runner = DetectorRunner()

    def build(
        self,
        rows: Iterable[dict[str, str]],
        *,
        text_column: str,
        span_text_column: str,
        span_start_column: str,
        span_end_column: str,
        pii_type_column: str,
        label_column: str,
    ) -> tuple[np.ndarray, list[str]]:
        features_list: list[np.ndarray] = []
        pre_processed_rows = []
        labels: list[str] = []
        for row in rows:
            text = row.get(text_column, "")
            if not text:
                LOG.warning("Skipping row with empty text: %s", row)
                continue
            # Support nested span object if present
            if "span" in row and isinstance(row["span"], dict):
                span_start_str = row["span"].get("start")
                span_end_str = row["span"].get("end")
                span_text_val = row["span"].get("text")
                pii_type_val = row["span"].get("pii_type") or row.get(pii_type_column)
            else:
                span_start_str = row.get(span_start_column)
                span_end_str = row.get(span_end_column)
                span_text_val = row.get(span_text_column)
                pii_type_val = row.get(pii_type_column, "OTHER")
                confidence_val = float(row.get("confidence", 0.8))
                sources_val = list(row.get("sources", []) or [])

            if span_start_str is None or span_end_str is None:
                # --- ROBUST DATA REPAIR LOGIC ---
                span_start, span_end = -1, -1
                # 1. First, try to find the provided (but often incorrect) span_text in the text.
                if span_text_val and span_text_val in text:
                    span_start = text.find(span_text_val)
                    span_end = span_start + len(span_text_val)

                # 2. If not found, fall back to running detectors to find a span of the correct type.
                if span_start == -1:
                    detected_spans = self.repair_runner.run(text)
                    found_span = None
                    for det_span in detected_spans:
                        if det_span.pii_type == pii_type_val:
                            found_span = det_span
                            break

                    if not found_span:
                        LOG.warning("Skipping row: Could not find or detect a span for pii_type '%s' in text: '%s...'", pii_type_val, text[:100])
                        continue

                    span_start = found_span.start
                    span_end = found_span.end
                    span_text = found_span.text
                    confidence_val = getattr(found_span, "confidence", 0.8)
                    sources_val = [getattr(found_span, "detector_name", "detector")]
                else:
                    span_text = span_text_val
            else:
                span_start = int(float(span_start_str))
                span_end = int(float(span_end_str))
                span_text = span_text_val or text[span_start:span_end]

            pii_type = pii_type_val or "OTHER"
            label_value = row.get(label_column)
            if label_value is None:
                LOG.warning("Skipping row with missing label: %s", row)
                continue
            label = normalise_label(label_value)
            if label not in CLASS_LIST:
                LOG.warning("Skipping row with invalid label '%s': %s", label, row)
                continue

            pre_processed_rows.append(
                (text, int(span_start), int(span_end), span_text, pii_type, float(confidence_val), list(sources_val))
            )
            labels.append(label)

        if not pre_processed_rows:
            raise ValueError("No valid training rows found.")

        # Batch process all embeddings for efficiency
        LOG.info("Building spans and context for batch embedding...")
        spans = [
            self._build_span(text, start, end, span_text, pii_type, confidence, sources)
            for text, start, end, span_text, pii_type, confidence, sources in pre_processed_rows
        ]
        # Compute neighbor stats per text to mirror inference-time features.
        neighbor_stats: dict[int, tuple[int, int]] = {}
        text_to_indices: dict[str, list[int]] = {}
        for idx, (text, *_rest) in enumerate(pre_processed_rows):
            text_to_indices.setdefault(text, []).append(idx)
        for text, indices in text_to_indices.items():
            local_spans = [spans[i] for i in indices]
            local_stats = compute_neighbor_stats(local_spans)
            for local_idx, stats in local_stats.items():
                neighbor_stats[indices[local_idx]] = stats
        span_texts = [s.text for s in spans]
        context_texts = [
            get_span_context_text(row[0], row[1], row[2]) for row in pre_processed_rows
        ]

        LOG.info("Encoding %d span embeddings in a batch...", len(span_texts))
        span_embeddings = self.embedding_model.encode_batch(span_texts)
        LOG.info("Encoding %d context embeddings in a batch...", len(context_texts))
        context_embeddings = self.embedding_model.encode_batch(context_texts)

        LOG.info("Extracting features for all rows...")
        for i, span in enumerate(spans):
            text = pre_processed_rows[i][0]
            sentences = compute_sentence_boundaries(text)
            sentence_index, sentence_ratio = compute_span_sentence_context(span, sentences)
            token_ratio = compute_token_position_ratio(text, span)
            neighbor_span_count, neighbor_high_risk_count = neighbor_stats.get(i, (0, 0))

            features = extract_span_features(
                span,
                text,
                embedding=span_embeddings[i],
                context_embedding=context_embeddings[i],
                sentence_index=sentence_index,
                sentence_position_ratio=sentence_ratio,
                token_position_ratio=token_ratio,
                neighbor_span_count=neighbor_span_count,
                neighbor_high_risk_count=neighbor_high_risk_count,
            )
            vector = span_features_to_vector(features)
            if vector.shape[0] != len(FEATURE_NAMES):
                raise AssertionError(
                    f"Feature vector length mismatch: expected {len(FEATURE_NAMES)}, got {vector.shape[0]}"
                )
            features_list.append(vector)
        if not features_list:
            raise ValueError("No valid training rows found.")
        return np.vstack(features_list), labels

    def _build_span(
        self,
        text: str,
        start: int,
        end: int,
        span_text: str,
        pii_type: str,
        confidence: float,
        sources: list[str],
    ) -> FusedSpan:
        span = FusedSpan(
            start=start,
            end=end,
            text=span_text,
            pii_type=pii_type,
            max_confidence=confidence,
            sources=list(sources),
        )
        if not self.use_detectors or self.runner is None or self.fusion is None:
            return span
        matched = self._match_detector_span(text, start, end, iou_threshold=0.8)
        if matched:
            span = FusedSpan(
                start=start,
                end=end,
                text=span_text or matched.text,
                pii_type=matched.pii_type or pii_type,
                max_confidence=getattr(matched, "max_confidence", 0.8),
                sources=matched.sources,
            )
        return span

    @lru_cache(maxsize=64)
    def _run_detectors(self, text: str) -> tuple[FusedSpan, ...]:
        assert self.runner and self.fusion
        results = self.runner.run(text)
        fused = self.fusion.fuse(results, text)
        return tuple(fused)

    def _match_detector_span(
        self, text: str, start: int, end: int, iou_threshold: float
    ) -> FusedSpan | None:
        """Find the best detector span match based on IoU."""
        detected_spans = self._run_detectors(text)
        best_match = None
        max_iou = 0.0
        for det_span in detected_spans:
            # Calculate intersection and union
            inter_start = max(start, det_span.start)
            inter_end = min(end, det_span.end)
            intersection = max(0, inter_end - inter_start)
            if intersection == 0:
                continue

            union = (end - start) + (det_span.end - det_span.start) - intersection
            iou = intersection / union if union > 0 else 0

            if iou > max_iou:
                max_iou = iou
                best_match = det_span

        if max_iou >= iou_threshold:
            return best_match
        return None


def normalise_label(raw_value: str | float) -> str:
    # Handle cases where pandas has already inferred a numeric type
    if isinstance(raw_value, (float, int)):
        numeric = max(0.0, min(1.0, float(raw_value)))
        # This needs to access a class method on SeverityModel, which is awkward here.
        # Re-implementing the logic is safer.
        if numeric >= 0.85:
            return "VERY_HIGH"
        if numeric >= 0.6:
            return "HIGH"
        if numeric >= 0.2:
            return "MEDIUM"
        return "LOW"

    # Handle string inputs
    value = str(raw_value).strip()
    try:
        numeric = float(value)
        numeric = max(0.0, min(1.0, numeric))
        if numeric >= 0.85:
            return "VERY_HIGH"
        if numeric >= 0.6:
            return "HIGH"
        if numeric >= 0.2:
            return "MEDIUM"
        return "LOW"
    except (ValueError, TypeError):
        # This will catch cases where the string is not a number, e.g., "HIGH"
        pass
    return LABEL_MAP.get(value.upper(), "LOW")


def train(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    # Deterministic seeds where possible
    try:
        import random
        random.seed(args.random_seed)
    except Exception:
        pass
    np.random.seed(args.random_seed)

    LOG.info("Loading and de-duplicating dataset from %s", args.input)
    if args.input.suffix.lower() == ".jsonl":
        df = pd.DataFrame(load_rows(args.input))
    else:
        df = pd.read_csv(args.input)
    initial_count = len(df)

    # De-duplicate based on the content that defines a unique sample.
    # This prevents the same example from appearing in both train and validation sets.
    unique_columns = [args.text_column]
    if args.span_start_column in df.columns and args.span_end_column in df.columns:
        unique_columns.extend([args.span_start_column, args.span_end_column])
    elif args.span_text_column in df.columns:
        LOG.warning("De-duplicating on 'text' and 'span_text' as start/end columns are missing.")
        unique_columns.append(args.span_text_column)

    df.drop_duplicates(subset=unique_columns, keep="first", inplace=True)
    final_count = len(df)
    LOG.info("Removed %d duplicate rows. Using %d unique training samples.", initial_count - final_count, final_count)

    rows = df.to_dict("records")

    builder = SpanDatasetBuilder(use_detectors=not args.disable_detector_augmentation)
    X, y = builder.build(
        rows,
        text_column=args.text_column,
        span_text_column=args.span_text_column,
        span_start_column=args.span_start_column,
        span_end_column=args.span_end_column,
        pii_type_column=args.pii_type_column,
        label_column=args.label_column,
    )
    if X.shape[1] != len(FEATURE_NAMES):
        raise AssertionError(f"Expected {len(FEATURE_NAMES)} features, got {X.shape[1]}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.validation_split_size,
        random_state=args.random_seed,
        stratify=y,
    )

    # Use a LightGBM model with optional calibration
    num_classes = len(set(y))
    LOG.info("Training samples: %d | Classes: %s", len(y), {c: y.count(c) for c in set(y)})
    lgbm_params = {
        "objective": "multiclass",
        "class_weight": "balanced",
        "random_state": args.random_seed,
        "num_class": max(2, num_classes),
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
    }
    try:
        import torch

        if torch.cuda.is_available():
            LOG.info("CUDA available. Attempting to use GPU for LightGBM training.")
            lgbm_params["device"] = "gpu"
    except ImportError:
        LOG.warning("torch not found, cannot auto-detect GPU. Defaulting to CPU for LightGBM.")

    base_model = LGBMClassifier(**lgbm_params)

    LOG.info("Training span-level LightGBM model...")
    X_train_df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    X_val_df = pd.DataFrame(X_val, columns=FEATURE_NAMES)
    try:
        base_model.fit(X_train_df, y_train)
    except Exception as e:
        if "GPU" in str(e) and lgbm_params.get("device") == "gpu":
            LOG.warning("LightGBM GPU training failed: %s. Falling back to CPU.", e)
            lgbm_params.pop("device")
            base_model = LGBMClassifier(**lgbm_params)
            base_model.fit(X_train_df, y_train)
        else:
            raise

    model = base_model
    if args.calibration_method != "none":
        if len(set(y_val)) < 2 or len(y_val) < 3:
            LOG.warning("Skipping calibration due to single-class validation set.")
        else:
            LOG.info("Calibrating probabilities using %s on validation split.", args.calibration_method)
            calibrator = CalibratedClassifierCV(
                estimator=base_model,
                method=args.calibration_method,
                cv="prefit",
            )
            try:
                calibrator.fit(X_val_df, y_val)
                model = calibrator
            except Exception as exc:
                LOG.warning("Calibration failed (%s). Using uncalibrated model.", exc)

    LOG.info("Evaluating validation performance.")
    y_pred = model.predict(X_val_df)
    report = classification_report(y_val, y_pred, labels=CLASS_LIST, zero_division=0)
    matrix = confusion_matrix(y_val, y_pred, labels=CLASS_LIST)
    LOG.info("Validation classification report:\n%s", report)
    LOG.info("Validation confusion matrix:\n%s", matrix)

    # Cross-validation for stability (without calibration to reduce cost)
    skf = StratifiedKFold(n_splits=max(2, args.cv_folds), shuffle=True, random_state=args.random_seed)
    cv_scores = []
    for train_idx, test_idx in skf.split(X, y):
        fold_model = LGBMClassifier(**lgbm_params)
        fold_model.fit(pd.DataFrame(X[train_idx], columns=FEATURE_NAMES), [y[i] for i in train_idx])
        preds = fold_model.predict(pd.DataFrame(X[test_idx], columns=FEATURE_NAMES))
        fold_report = classification_report([y[i] for i in test_idx], preds, output_dict=True, zero_division=0)
        cv_scores.append(fold_report)
    LOG.info("Cross-validation completed with %d folds.", len(cv_scores))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    # Persist feature names for downstream schema checks.
    try:
        model.feature_names_in_ = np.array(FEATURE_NAMES)  # type: ignore[attr-defined]
    except Exception:
        model.pat_feature_names = FEATURE_NAMES  # fallback when property is read-only
    try:
        model.schema_version = FEATURE_SCHEMA_VERSION  # type: ignore[attr-defined]
    except Exception:
        ...
    try:
        model.class_labels = CLASS_LIST  # type: ignore[attr-defined]
    except Exception:
        ...
    joblib.dump(model, args.output)  # type: ignore[arg-type]
    metadata = {
        "feature_names": FEATURE_NAMES,
        "class_labels": CLASS_LIST,
        "schema_version": FEATURE_SCHEMA_VERSION,
    }
    meta_path = args.output.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOG.info("Model saved to %s (metadata -> %s)", args.output, meta_path)

    # Save feature importance and training stats
    if hasattr(base_model, "feature_importances_"):
        importance = {name: float(val) for name, val in zip(FEATURE_NAMES, base_model.feature_importances_)}
    else:
        importance = {name: 0.0 for name in FEATURE_NAMES}
    importance_path = args.report_dir / f"{args.output.stem}_feature_importance.json"
    importance_path.write_text(json.dumps(importance, indent=2), encoding="utf-8")

    train_stats = {}
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    for idx, name in enumerate(FEATURE_NAMES):
        train_stats[name] = {
            "mean": float(means[idx]),
            "std": float(stds[idx]),
            "min": float(mins[idx]),
            "max": float(maxs[idx]),
        }
    stats_path = args.report_dir / f"{args.output.stem}_training_stats.json"
    stats_path.write_text(json.dumps(train_stats, indent=2), encoding="utf-8")
    LOG.info("Feature importance -> %s | Training stats -> %s", importance_path, stats_path)


if __name__ == "__main__":  # pragma: no cover
    train()
            
