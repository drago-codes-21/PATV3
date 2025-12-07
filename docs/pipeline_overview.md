# Pipeline Overview (PAT)

Flow: text → detectors (regex, ml_ner, domain_heuristic, semantic, ml_token) → fusion (non-overlapping FusedSpans) → severity (feature builder + model) → policy (rules + masking) → sanitized text.

Key components:
- `pat.pipeline.core.run_pipeline`: single entry to run end-to-end and return structured `PipelineResult`.
- `pat.pipeline.pipeline.RedactionPipeline`: orchestrates detectors, fusion, severity, policy.
- `pat.pipeline.run`: CLI for file input/output.
- `pat.pipeline.debug`: developer harness to inspect spans/decisions.

Inputs:
- Plain text strings (CLI reads from file).
- Optional context dict (`channel`, `section`, etc.) passed into severity/policy.

Outputs:
- `sanitized_text`, decision, severity summary, fused spans (with detector evidence, severity, policy decisions), raw detector results.

Config/Models:
- Settings via `pat.config.settings.get_settings`.
- Detector configs: regex patterns, heuristics, thresholds.
- Severity: model path/config (`PAT_SEVERITY_MODEL_PATH` override), thresholds in settings.
- Policy: `pat/config/policy_rules.json` (rule schema described in docs/policy_engine.md).

Debugging:
- Env flags `PAT_DEBUG_DETECTORS`, `PAT_DEBUG_SEVERITY`, `PAT_DEBUG_POLICY`, `PAT_DEBUG_FUSION`.
- `python -m pat.pipeline.debug --text "..."`
