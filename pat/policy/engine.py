"""PII Redaction Policy Engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from pat.config import get_settings
from pat.policy.masking import apply_mask
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
            severity_label = self.get_severity_label(span_severity_score or 0.0)
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
                action_config = rule.get("action", {})
                action_type = action_config.get("decision")
                if not action_type:
                    continue

                placeholder = self._get_placeholder(span, action_config)
                setattr(
                    span,
                    "policy_decision",
                    {
                        "decision": action_type,
                        "rule_id": rule.get("id"),
                        "placeholder": placeholder,
                        "severity_label": severity_label,
                    },
                )
                decisions.append(
                    PolicyDecision(
                        span=span,
                        action=action_type,
                        rule_id=rule.get("id"),
                        placeholder=placeholder,
                        severity_label=severity_label,
                    )
                )
                matched_rule = True
                break  # Move to the next span

            if not matched_rule:
                setattr(
                    span,
                    "policy_decision",
                    {
                        "decision": "ALLOW",
                        "rule_id": "default_allow",
                        "placeholder": None,
                        "severity_label": severity_label,
                    },
                )
                decisions.append(
                    PolicyDecision(
                        span=span,
                        action="ALLOW",
                        rule_id="default_allow",
                        severity_label=severity_label,
                    )
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
            return mask_args.get("placeholder")

        # Fallback for other strategies or legacy formats, though the provided JSON uses 'placeholder'.
        if "style" in action_config:
            return apply_mask(action_config["style"], span.text, action_config)

        return None
