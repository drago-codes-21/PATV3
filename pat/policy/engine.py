"""PII Redaction Policy Engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pat.config import get_settings
from pat.policy.masking import apply_mask
from pat.utils.taxonomy import placeholder_for_type
from pat.fusion import FusedSpan

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyDecision:
    """The output of a policy decision for a single span."""

    span: FusedSpan
    action: str  # e.g., MASK, BLOCK, ALLOW
    rule_id: str | None
    placeholder: str | None = None
    severity_label: str | None = None
    masked_text: str | None = None
    mask_strategy: str | None = None
    mask_args: dict[str, Any] | None = None


class PolicyEngine:
    """Loads and evaluates redaction policies from a YAML file."""

    def __init__(self, policy_rules_path: Path | None = None) -> None:
        """Initialise the engine by loading policy rules."""
        settings = get_settings()
        self.path = policy_rules_path or settings.policy_file_path
        if not self.path.exists():
            raise FileNotFoundError(f"Policy file not found at {self.path}")

        with self.path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

        cfg_thresholds = self.config.get("severity_thresholds") or settings.severity_thresholds
        if not cfg_thresholds:
            LOG.warning(
                "No severity thresholds present in policy config. "
                "Falling back to application defaults."
            )
            cfg_thresholds = settings.severity_thresholds
        # Normalise thresholds to floats and keep a deterministic ordering by lower bound.
        self.severity_thresholds: dict[str, tuple[float, float]] = {
            label: (float(bounds[0]), float(bounds[1])) for label, bounds in cfg_thresholds.items()
        }
        ordered_labels = sorted(self.severity_thresholds.items(), key=lambda item: item[1][0])
        self.severity_labels = [label for label, _ in ordered_labels]
        self.severity_map = {label: i for i, label in enumerate(self.severity_labels)}

        # Respect rule priority (high first) for deterministic outcomes.
        self.rules = sorted(
            self.config.get("rules", []),
            key=lambda rule: rule.get("priority", 0),
            reverse=True,
        )

    def get_severity_label(self, score: float) -> str:
        """Map a numeric score to a severity label from the policy."""
        for label, (low, high) in sorted(
            self.severity_thresholds.items(), key=lambda item: item[1][0]
        ):
            if low <= score <= high:
                return label
        # If the score falls outside configured ranges, treat it as highest risk.
        return self.severity_labels[-1] if self.severity_labels else "UNKNOWN"

    def decide(
        self, spans: list[FusedSpan], context: dict
    ) -> list[PolicyDecision]:
        """
        Evaluate policy rules against detected spans and context.

        Returns a list of redaction decisions, one for each span that should be acted upon.
        """
        decisions: list[PolicyDecision] = []

        for span in spans:
            span_severity_score = getattr(span, "severity_score", 0.0)
            severity_label = getattr(span, "severity_label", None) or self.get_severity_label(
                span_severity_score or 0.0
            )
            matched_rule = False

            for rule in self.rules:
                if not rule.get("enabled", True):
                    continue

                # Check severity condition for this specific span
                min_sev = rule.get("severity_at_least")
                if min_sev and self.severity_map.get(severity_label, -1) < self.severity_map.get(
                    min_sev, 99
                ):
                    continue

                sev_in = rule.get("severity_in")
                if sev_in and severity_label not in sev_in:
                    continue

                # Check PII type condition for this specific span
                pii_any = rule.get("pii_types")
                if pii_any and span.pii_type not in pii_any:
                    continue

                # First matching rule wins
                action_config: dict[str, Any] = rule.get("action", {}) or {}
                action_type = action_config.get("decision")
                if not action_type:
                    continue

                mask_strategy = action_config.get("mask_strategy") or "placeholder"
                mask_args = action_config.get("mask_args", {}) or {}
                placeholder = self._get_placeholder(span, action_config)
                masked_text = (
                    apply_mask(mask_strategy, span.text, mask_args)
                    if action_type == "MASK"
                    else None
                )
                setattr(
                    span,
                    "policy_decision",
                    {
                        "decision": action_type,
                        "rule_id": rule.get("id"),
                        "placeholder": placeholder,
                        "severity_label": severity_label,
                        "mask_strategy": mask_strategy,
                        "mask_args": mask_args,
                    },
                )
                decisions.append(
                    PolicyDecision(
                        span=span,
                        action=action_type,
                        rule_id=rule.get("id"),
                        placeholder=placeholder,
                        severity_label=severity_label,
                        masked_text=masked_text,
                        mask_strategy=mask_strategy,
                        mask_args=mask_args,
                    )
                )
                matched_rule = True
                break  # Move to the next span

            if not matched_rule:
                # Default decision is severity-aware: never allow VERY_HIGH, and only
                # allow HIGH when explicitly configured. Defaults are conservative.
                default_action = "ALLOW"
                default_rule = "default_allow"
                placeholder = None
                masked_text = None
                if severity_label in {"HIGH", "VERY_HIGH"}:
                    default_action = "MASK"
                    default_rule = "default_mask_high"
                    placeholder = self._get_placeholder(span, {"mask_strategy": "placeholder"})
                    masked_text = apply_mask("placeholder", span.text, {"placeholder": placeholder})

                setattr(
                    span,
                    "policy_decision",
                    {
                        "decision": default_action,
                        "rule_id": default_rule,
                        "placeholder": placeholder,
                        "severity_label": severity_label,
                    },
                )
                decisions.append(
                    PolicyDecision(
                        span=span,
                        action=default_action,
                        rule_id=default_rule,
                        severity_label=severity_label,
                        placeholder=placeholder,
                        masked_text=masked_text,
                    )
                )

            # Safety invariant: VERY_HIGH spans must never be allowed.
            last_decision = decisions[-1]
            if severity_label == "VERY_HIGH" and last_decision.action == "ALLOW":
                safe_placeholder = last_decision.placeholder or self._get_placeholder(
                    span, {"mask_strategy": "placeholder"}
                )
                safe_masked = apply_mask("placeholder", span.text, {"placeholder": safe_placeholder})
                safe = PolicyDecision(
                    span=span,
                    action="MASK",
                    rule_id=f"{last_decision.rule_id}_forced_mask" if last_decision.rule_id else "forced_mask",
                    placeholder=safe_placeholder,
                    severity_label=severity_label,
                    masked_text=safe_masked,
                    mask_strategy="placeholder",
                    mask_args={"placeholder": safe_placeholder},
                )
                decisions[-1] = safe
                setattr(
                    span,
                    "policy_decision",
                    {
                        "decision": safe.action,
                        "rule_id": safe.rule_id,
                        "placeholder": safe.placeholder,
                        "severity_label": safe.severity_label,
                        "mask_strategy": safe.mask_strategy,
                        "mask_args": safe.mask_args,
                    },
                )

        return decisions

    def _get_placeholder(self, span: FusedSpan, action_config: dict) -> str | None:
        """Generate a placeholder for a given span and action."""
        action_type = action_config.get("decision")

        if action_type != "MASK":
            return None

        mask_strategy = action_config.get("mask_strategy")
        mask_args = action_config.get("mask_args", {})

        if mask_strategy == "placeholder":
            return mask_args.get("placeholder") or placeholder_for_type(span.pii_type)

        # Default to a taxonomy placeholder if available when no explicit strategy is provided.
        placeholder = placeholder_for_type(span.pii_type)
        if placeholder:
            return placeholder

        # Fallback for other strategies or legacy formats, though the provided JSON uses 'placeholder'.
        if "style" in action_config:
            return apply_mask(action_config["style"], span.text, action_config)

        return f"<{span.pii_type}>"
