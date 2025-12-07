# PAT Detector Calibration + Severity Stability

This project contains the PAT V4 detector → fusion → severity pipeline. Key maintenance tasks:

- **Schema lock:** Severity features are fixed at 41 dimensions (`pat/severity/features.py`). The schema is asserted in code and tests (`tests/test_severity_feature_schema.py`). Any change must bump the schema intentionally.
- **Analysis and drift:** Use `python -m pat.severity.analysis --input spans.jsonl --model model.joblib --train-stats model_training_stats.json` to view feature importance and drift.
- **Threshold calibration:** Run `python -m pat.calibration.calibrate_thresholds --data labeled_spans.jsonl --output thresholds.json` to sweep MPNet embedding and ML token thresholds for best F1.
- **Severity training:** Train with `python -m pat.severity.train --input training_data.jsonl --output model.joblib`. Outputs include feature importance and training stats alongside the model; cross-validation is built in.
- **E2E evaluation:** Execute `python -m pat.evaluation.run_e2e_report --input test_corpus.jsonl` to generate `e2e_report.md/json` with precision/recall, boundary IoU, detector agreement, and severity distributions.
- **Validation harness:** `python tests/detector_layer_validation.py` runs synthetic bug regressions (adjacent bank+email, punctuated phones, messy credentials, long IBANs) and asserts trimmed spans and fusion separation.
- **Debug logging:** Set `PAT_DEBUG_DETECTORS=true` to emit JSONL debug events for near-threshold detectors, fusion boundary/type changes, and ambiguous severity scores.
- **Continuous Improvement Pipeline (CIP):** Run `python -m pat.cip.orchestrator --config pat/config/settings.yaml` to execute regression tests, optional threshold calibration, drift checks, severity retraining, and end-to-end evaluation. Artifacts (reports, thresholds, models) are written to `pat/cip/artifacts/`.
- **Policy Engine:** Masking rules live in `pat/config/policy_rules.json`. The policy engine applies severity-aware masking styles (full, preserve-format, partial email, preserve last N) after fusion and severity scoring. CLI: `python -m pat.pipeline.run --input email.txt --output sanitized.txt --debug`.
- **Severity Training:** Train with `python -m pat.severity.train --input data.jsonl --output pat/models/severity/severity_model_latest.joblib --report-dir pat/models/severity/reports/`. Input JSONL/CSV must include `text`, `span_start`, `span_end`, `pii_type`, and `severity_label` (optionally nested under `span`). Reports include feature importance and training stats.
- **Severity Inference:** Run severity-only inspection with `python -m pat.severity.infer --text "My email is test@example.com"` (or `--input file.txt`, `--json-output out.json`). Outputs detected spans with severity labels/scores. Enable `PAT_DEBUG_SEVERITY=true` (and `--debug`) to include feature dumps; detector debug via `PAT_DEBUG_DETECTORS`.
- **Pipeline Flow:** raw text → detectors → fusion → 41-dim severity features → severity model → policy engine → sanitized text.

All utilities are offline-friendly and use the canonical MPNet embedding stack.
