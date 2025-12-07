from pathlib import Path
import json

from pat.fusion import FusedSpan
from pat.policy.engine import PolicyEngine
from pat.pipeline.pipeline import _apply_redactions

# Content from pat/config/policy_rules.json for self-contained tests
POLICY_JSON_CONTENT = """
{
  "severity_thresholds": {
    "LOW": [0.0, 0.2],
    "MEDIUM": [0.2, 0.6],
    "HIGH": [0.6, 0.85],
    "VERY_HIGH": [0.85, 1.0]
  },
  "rules": [
    {
      "id": "financial_high",
      "pii_types": ["BANK_ACCOUNT", "CARD_NUMBER", "SORT_CODE"],
      "severity_at_least": "LOW",
      "action": { "decision": "MASK", "mask_strategy": "placeholder", "mask_args": { "placeholder": "<FINANCIAL>" } },
      "enabled": true
    },
    {
      "id": "contact_medium",
      "pii_types": ["EMAIL", "PHONE"],
      "severity_at_least": "MEDIUM",
      "action": { "decision": "MASK", "mask_strategy": "placeholder", "mask_args": { "placeholder": "<CONTACT>" } },
      "enabled": true
    },
    {
      "id": "contact_low_allow",
      "pii_types": ["EMAIL", "PHONE"],
      "severity_in": ["LOW"],
      "action": { "decision": "ALLOW" },
      "enabled": true
    }
  ]
}
"""

def span(start: int, end: int, text: str, pii_type: str, confidence: float = 0.9) -> FusedSpan:
    """Helper to create a FusedSpan for testing."""
    return FusedSpan(
        start=start,
        end=end,
        text=text[start:end],
        pii_type=pii_type,
        max_confidence=confidence,
        sources=["test_detector"],
    )


def get_test_engine(tmp_path: Path) -> PolicyEngine:
    """Writes the test policy to a file and returns an engine instance."""
    policy_path = tmp_path / "test_policy.json"
    policy_path.write_text(POLICY_JSON_CONTENT, encoding="utf-8")
    # The PolicyEngine uses yaml.safe_load, which can parse JSON.
    return PolicyEngine(policy_rules_path=policy_path)


def test_financial_masking(tmp_path: Path):
    """Verify a BANK_ACCOUNT is masked as <FINANCIAL> for any severity >= LOW."""
    engine = get_test_engine(tmp_path)
    text = "My account number is 12345678."
    s = span(text.find("12345678"), text.find("12345678") + 8, text, "BANK_ACCOUNT")

    # Manually set the severity on the span object for the test
    s.severity_score = 0.5  # MEDIUM score
    decisions = engine.decide([s], context={})
    sanitized_text = _apply_redactions(text, decisions)

    assert "12345678" not in sanitized_text
    assert "<FINANCIAL>" in sanitized_text


def test_contact_mask_medium_severity(tmp_path: Path):
    """Verify a MEDIUM severity EMAIL gets masked as <CONTACT>."""
    engine = get_test_engine(tmp_path)
    text = "My email is user@example.com."
    s = span(text.find("user@example.com"), text.find("user@example.com") + len("user@example.com"), text, "EMAIL")

    # Manually set the severity on the span object for the test
    s.severity_score = 0.5  # MEDIUM score
    decisions = engine.decide([s], context={})
    sanitized_text = _apply_redactions(text, decisions)

    assert "user@example.com" not in sanitized_text
    assert "<CONTACT>" in sanitized_text


def test_contact_allow_low_severity(tmp_path: Path):
    """Verify a LOW severity EMAIL is allowed."""
    engine = get_test_engine(tmp_path)
    text = "My email is user@example.com."
    s = span(text.find("user@example.com"), text.find("user@example.com") + len("user@example.com"), text, "EMAIL")

    # Manually set the severity on the span object for the test
    s.severity_score = 0.1  # LOW score
    decisions = engine.decide([s], context={})
    sanitized_text = _apply_redactions(text, decisions)

    assert len(decisions) == 1
    assert decisions[0].action == "ALLOW"
    assert decisions[0].rule_id == "contact_low_allow"
    assert "user@example.com" in sanitized_text


def test_default_allow_unmatched_pii(tmp_path: Path):
    """Verify that a LOW severity PERSON span is allowed by default as no rule matches."""
    engine = get_test_engine(tmp_path)
    text = "I spoke with John Doe."
    s = span(text.find("John Doe"), text.find("John Doe") + len("John Doe"), text, "PERSON")

    # Manually set the severity on the span object for the test
    s.severity_score = 0.1  # LOW score
    decisions = engine.decide([s], context={})
    sanitized_text = _apply_redactions(text, decisions)

    assert len(decisions) == 1
    assert decisions[0].action == "ALLOW"
    assert decisions[0].rule_id == "default_allow"
    assert "John Doe" in sanitized_text


def test_policy_engine_defaults_thresholds_when_missing(tmp_path: Path):
    """Ensure missing severity_thresholds fall back to application defaults."""
    policy_path = tmp_path / "policy_no_thresholds.json"
    policy_data = {
        "rules": [
            {
                "id": "financial_mask",
                "pii_types": ["BANK_ACCOUNT"],
                "severity_at_least": "LOW",
                "action": {
                    "decision": "MASK",
                    "mask_strategy": "placeholder",
                    "mask_args": {"placeholder": "<FINANCIAL>"},
                },
                "enabled": True,
            }
        ]
    }
    policy_path.write_text(json.dumps(policy_data), encoding="utf-8")
    engine = PolicyEngine(policy_rules_path=policy_path)

    text = "Account number 12345678."
    s = span(text.find("12345678"), text.find("12345678") + 8, text, "BANK_ACCOUNT")
    s.severity_score = 0.0  # No model score available

    decisions = engine.decide([s], context={})
    assert decisions[0].action == "MASK"
    assert decisions[0].severity_label in {"LOW", "MEDIUM", "HIGH", "VERY_HIGH"}
    # High scores should still map to the topmost label
    assert engine.get_severity_label(1.5) in {"VERY_HIGH"}
