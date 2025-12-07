# Severity Layer Overview

## Components
- **Feature builder**: `pat.severity.features.extract_span_features` converts `FusedSpan` + document context into an ordered vector (`FEATURE_NAMES`, 87 dims). Categories: detector evidence (sources, scores, detector counts), span shape (ratios, flags, segments), context keywords/channels, neighbor density, embeddings, and PII-type one-hots (including GOV_ID, IP_ADDRESS, URL, GENERIC_NUMBER).
- **Model**: `SeverityModel` wraps a multi-class classifier (LightGBM by default) with optional probability calibration. Methods:
  - `predict(feature_vector, pii_type=None) -> (score, label, probs)`
  - Backward-compatible `predict_span_score/label` call `predict`.
  - Uses configurable probability thresholds (per-type overrides supported in settings) plus score buckets for safety.
- **Training**: `pat.severity.train` trains a LightGBM multi-class model, optionally calibrates (`--calibration-method [sigmoid|isotonic|none]`), saves feature schema, importance, and training stats. Hyperparameters (learning rate, num leaves, depth, estimators) are CLI-configurable.
- **Inference wiring**: Pipeline and `severity.infer` compute neighbor stats, build features, call `SeverityModel.predict`, and attach `severity_label`, `severity_score`, and `severity_probs` to each `FusedSpan`.

## Config/Debug
- Severity thresholds in `pat.config.settings`: `severity_thresholds`, `severity_type_thresholds`, `severity_probability_thresholds`.
- Debug logging: `PAT_DEBUG_SEVERITY=1` logs per-span predictions (label, score, probs) via `log_decision`.

## Tests
- Feature schema/length, keyword/neighbor features, inference/pipeline regression, and training smoke tests cover the severity path end-to-end.
