import json

from pat.fusion import FusedSpan
from pat.policy.engine import PolicyEngine
from pat.policy.masking import apply_mask
from pat.pipeline.pipeline import _apply_redactions


def _write_policy(path, rules):
    payload = {
        "severity_thresholds": {
            "LOW": [0.0, 0.2],
            "MEDIUM": [0.2, 0.6],
            "HIGH": [0.6, 0.85],
            "VERY_HIGH": [0.85, 1.0],
        },
        "rules": rules,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_policy_forces_mask_on_very_high(tmp_path):
    policy_path = tmp_path / "policy.json"
    _write_policy(
        policy_path,
        [
            {
                "id": "allow_email",
                "pii_types": ["EMAIL"],
                "severity_in": ["VERY_HIGH"],
                "action": {"decision": "ALLOW"},
                "enabled": True,
            }
        ],
    )
    engine = PolicyEngine(policy_rules_path=policy_path)
    text = "email user@example.com"
    span = FusedSpan(
        start=text.index("user"),
        end=text.index("user") + len("user@example.com"),
        text="user@example.com",
        pii_type="EMAIL",
        max_confidence=0.9,
        sources=["regex"],
    )
    span.severity_score = 0.95
    decisions = engine.decide([span], context={})
    assert decisions[0].action == "MASK"
    sanitized = _apply_redactions(text, decisions)
    assert "user@example.com" not in sanitized


def test_policy_applies_mask_strategy(tmp_path):
    policy_path = tmp_path / "policy.json"
    _write_policy(
        policy_path,
        [
            {
                "id": "format_mask",
                "pii_types": ["CARD_NUMBER"],
                "severity_at_least": "LOW",
                "action": {
                    "decision": "MASK",
                    "mask_strategy": "preserve_format",
                    "mask_args": {"mask_char": "X"},
                },
                "enabled": True,
            }
        ],
    )
    engine = PolicyEngine(policy_rules_path=policy_path)
    text = "Card 4111-1111-1111-1111"
    start = text.index("4111")
    span_text = "4111-1111-1111-1111"
    span = FusedSpan(
        start=start,
        end=start + len(span_text),
        text=span_text,
        pii_type="CARD_NUMBER",
        max_confidence=0.7,
        sources=["regex"],
    )
    span.severity_score = 0.4
    decisions = engine.decide([span], context={})
    masked_expected = apply_mask("preserve_format", span.text, {"mask_char": "X"})
    assert decisions[0].masked_text == masked_expected
    sanitized = _apply_redactions(text, decisions)
    assert masked_expected in sanitized
