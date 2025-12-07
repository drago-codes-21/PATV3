# Policy Engine Overview

## Rule Schema
- Rules live in `pat/config/policy_rules.json` as a list under `rules`.
- Each rule:
  - `id`, `description`, `enabled`, `priority`
  - Conditions (`pii_types`, `severity_at_least` or `severity_in`, optional `channels`, `context_keywords_any/all`, `min_prob`)
  - Action: `decision` (`MASK`, `ALLOW`, `DROP`, `ANNOTATE`), optional `mask_strategy`, `mask_args`
- Backward compatibility: legacy `{PII_TYPE: {...}}` maps are auto-migrated on load.

## Matching & Resolution
- For each `FusedSpan`, rules are filtered by type, severity, channel, context keywords, and probability thresholds.
- Conflict resolution: highest `priority` → most specific (more pii_types) → lexicographic `id`.
- If no rule matches: default behavior masks `HIGH/VERY_HIGH`, allows lower severities.
- Safety invariant: `VERY_HIGH` spans are forced to `MASK` even if a rule allows them.

## Masking Strategies
- Implemented in `pat.policy.masking.apply_mask`; strategies: `full`, `partial_email`, `preserve_last_n`, `preserve_format`, `mask_credential`, `numeric_token`, `placeholder`, `semantic_mask`.
- Type defaults (`DEFAULT_STRATEGY_BY_TYPE`) provide sensible fallbacks per PII type.
- Failures fall back to `full` mask.

## Integration
- `PolicyEngine.apply_policy(text, spans, context)` returns sanitized text and attaches `policy_decision` to spans.
- `PolicyEngine.decide` remains pipeline entrypoint; context (channel/section) is optional.
- Debugging: `PAT_DEBUG_POLICY=1` logs rule selection, decision, and masked spans.

## Invariants
- Non-overlapping spans expected (fusion layer). Overlaps are warned and skipped.
- VERY_HIGH severity is never left unmasked.
- Deterministic: tie-breaking is stable; same input/config yields same output.
