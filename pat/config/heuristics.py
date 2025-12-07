"""Configuration for heuristic, context-aware PII rules.

These rules are *contextual hints* layered on top of more precise detectors
(regex, ML, embeddings). They should bias towards precision to avoid
over-masking, and are treated as lower-confidence signals unless strongly
keyworded and numeric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Pattern, Sequence, Tuple


HEURISTIC_VERSION = "1.1.0"


@dataclass(frozen=True)
class HeuristicRule:
    """Declarative heuristic rule.

    Fields:
        id:
            Stable identifier for the rule, used in logs/analysis.
        pii_type:
            Logical PII type (must map into the system taxonomy).
        any_keywords:
            At least one of these must appear in the text for the rule to fire.
        all_keywords:
            If non-empty, *all* of these must appear somewhere in the text
            (global precondition).
        category:
            High-level category (e.g. FINANCIAL, CONTACT, HEALTH).
        context_window:
            Number of characters on each side of the keyword to search within.
        requires_digits:
            If True, we only emit spans from `number_pattern` matches.
        number_pattern:
            Optional numeric pattern; required if `requires_digits=True`.
        base_score:
            Baseline detector confidence; typically < regex detectors.
        right_context:
            For non-numeric rules, how far to expand to the right of the
            keyword before trimming (to avoid swallowing whole sentences).
        priority:
            Lower numbers evaluated earlier; used for deterministic ordering.
        allow_partial_window:
            Reserved for potential future use; currently informational.
        hint:
            Optional human-readable description (e.g. "financial", "health").
    """

    id: str
    pii_type: str
    any_keywords: Tuple[str, ...]
    all_keywords: Tuple[str, ...] = ()
    category: Optional[str] = None
    context_window: int = 64
    requires_digits: bool = False
    number_pattern: Optional[Pattern[str]] = None
    base_score: float = 0.6
    right_context: int = 16
    priority: int = 100
    allow_partial_window: bool = True
    hint: Optional[str] = None


HEURISTIC_RULES: Sequence[HeuristicRule] = (
    # ------------------------------------------------------------------ #
    # FINANCIAL
    # ------------------------------------------------------------------ #
    HeuristicRule(
        id="sort_code",
        pii_type="SORT_CODE",
        category="FINANCIAL",
        any_keywords=("sort code", "sort-code"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)(\d{2}[-\s]?\d{2}[-\s]?\d{2})(?!\d)"),
        base_score=0.75,
        hint="financial",
        priority=10,
    ),
    HeuristicRule(
        id="account_number",
        pii_type="BANK_ACCOUNT",
        category="FINANCIAL",
        any_keywords=("account number", "account no", "acct no", "iban"),
        requires_digits=True,
        # Slightly broad; account-number validator and fusion should refine.
        number_pattern=re.compile(r"(?<!\d)(\d{8,18})(?!\d)"),
        base_score=0.72,
        hint="financial",
        priority=15,
    ),
    HeuristicRule(
        id="card_number",
        pii_type="CARD_NUMBER",
        category="FINANCIAL",
        any_keywords=("card number", "debit card", "credit card", "pan"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
        base_score=0.78,
        hint="financial",
        priority=20,
    ),
    HeuristicRule(
        id="swift_routing",
        pii_type="BANK_ACCOUNT",
        category="FINANCIAL",
        any_keywords=("swift", "routing", "sort code", "ifsc"),
        requires_digits=True,
        number_pattern=re.compile(r"\b[A-Z]{4}\w{7}\b|\b\d{6,12}\b", flags=re.IGNORECASE),
        base_score=0.62,
        hint="financial",
        priority=30,
    ),
    # ------------------------------------------------------------------ #
    # CREDENTIALS / AUTH
    # ------------------------------------------------------------------ #
    HeuristicRule(
        id="credential_password",
        pii_type="CREDENTIAL",
        category="AUTHENTICATION",
        any_keywords=("password", "passcode", "pass code", "pwd", "passphrase"),
        requires_digits=False,
        base_score=0.85,
        right_context=24,
        hint="credential",
        priority=20,
    ),
    HeuristicRule(
        id="credential_token",
        pii_type="CREDENTIAL",
        category="AUTHENTICATION",
        any_keywords=("secret", "token", "api key", "api-key", "bearer token"),
        requires_digits=False,
        base_score=0.83,
        right_context=32,
        hint="credential",
        priority=25,
    ),
    HeuristicRule(
        id="credential_otp",
        pii_type="CREDENTIAL",
        category="AUTHENTICATION",
        any_keywords=("otp", "one-time passcode", "one time password", "verification code"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)\d{4,8}(?!\d)"),
        base_score=0.8,
        hint="credential",
        priority=22,
    ),
    HeuristicRule(
        id="credential_pin",
        pii_type="CREDENTIAL",
        category="AUTHENTICATION",
        any_keywords=("pin", "security number"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)\d{4,6}(?!\d)"),
        base_score=0.76,
        hint="credential",
        priority=22,
    ),
    # ------------------------------------------------------------------ #
    # GOVERNMENT / IDENTITY
    # ------------------------------------------------------------------ #
    HeuristicRule(
        id="gov_identity",
        pii_type="GOV_ID",
        category="GOVERNMENT_ID",
        any_keywords=("ssn", "social security", "passport", "driver", "license", "aadhaar", "pan"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)(\d{6,12})(?!\d)"),
        base_score=0.7,
        hint="identity",
        priority=30,
    ),
    HeuristicRule(
        id="nhs_number_heuristic",
        pii_type="NHS_NUMBER",
        category="GOVERNMENT_ID",
        any_keywords=("nhs number", "nhs"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)(\d{3}[\s-]?\d{3}[\s-]?\d{4})(?!\d)"),
        base_score=0.85,
        hint="identity",
        priority=25,
    ),
    # ------------------------------------------------------------------ #
    # CONTACT
    # ------------------------------------------------------------------ #
    HeuristicRule(
        id="contact_phone",
        pii_type="PHONE",
        category="CONTACT",
        any_keywords=("phone", "mobile", "cell", "whatsapp"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)(\+?\d[\d\-\s()]{6,}\d)(?!\d)"),
        base_score=0.68,
        hint="contact",
        priority=20,
    ),
    HeuristicRule(
        id="contact_address",
        pii_type="ADDRESS",
        category="CONTACT",
        any_keywords=("home address", "residence", "shipping address", "billing address"),
        requires_digits=False,
        base_score=0.65,
        right_context=48,
        hint="location",
        priority=40,
    ),
    # ------------------------------------------------------------------ #
    # HEALTH / MEDICAL CONTEXT
    # ------------------------------------------------------------------ #
    HeuristicRule(
        id="medical_context",
        pii_type="MEDICAL_INFO",
        category="HEALTH",
        any_keywords=("diagnosis", "diagnosed", "treatment", "symptom", "medical condition", "allergy"),
        requires_digits=False,
        base_score=0.7,
        right_context=32,
        hint="health",
        priority=35,
    ),
    HeuristicRule(
        id="medication_context",
        pii_type="MEDICATION",
        category="HEALTH",
        any_keywords=("medication", "medicine", "prescription", "rx", "tablet", "pill"),
        requires_digits=False,
        base_score=0.65,
        right_context=32,
        hint="health",
        priority=36,
    ),
    HeuristicRule(
        id="hospital_context",
        pii_type="HOSPITAL",
        category="HEALTH",
        any_keywords=("hospital", "clinic", "nhs trust"),
        requires_digits=False,
        base_score=0.62,
        right_context=32,
        hint="health",
        priority=38,
    ),
    # ------------------------------------------------------------------ #
    # EMPLOYMENT / EDUCATION
    # ------------------------------------------------------------------ #
    HeuristicRule(
        id="employment_employer",
        pii_type="EMPLOYER_NAME",
        category="EMPLOYMENT_EDUCATION",
        any_keywords=("employer", "company", "works at", "employed by"),
        requires_digits=False,
        base_score=0.6,
        right_context=48,
        hint="employment",
        priority=50,
    ),
    HeuristicRule(
        id="employment_id",
        pii_type="EMPLOYMENT_ID",
        category="EMPLOYMENT_EDUCATION",
        any_keywords=("employee id", "staff id", "payroll id"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\w)[A-Z]?\d{4,10}(?!\w)", flags=re.IGNORECASE),
        base_score=0.65,
        hint="employment",
        priority=45,
    ),
    HeuristicRule(
        id="job_title",
        pii_type="JOB_TITLE",
        category="EMPLOYMENT_EDUCATION",
        any_keywords=("job title", "role", "position"),
        requires_digits=False,
        base_score=0.55,
        right_context=48,
        hint="employment",
        priority=60,
    ),
    # ------------------------------------------------------------------ #
    # BIOMETRIC
    # ------------------------------------------------------------------ #
    HeuristicRule(
        id="biometric_context",
        pii_type="BIOMETRIC",
        category="BIOMETRIC",
        any_keywords=("fingerprint", "face id", "voiceprint", "iris scan", "retina"),
        requires_digits=False,
        base_score=0.72,
        right_context=24,
        hint="biometric",
        priority=30,
    ),
    # ------------------------------------------------------------------ #
    # DEVICE / TECHNICAL IDs
    # ------------------------------------------------------------------ #
    HeuristicRule(
        id="device_id",
        pii_type="DEVICE_ID",
        category="ONLINE_DEVICE",
        any_keywords=("device id", "imei", "udid", "android id"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\w)[A-Fa-f0-9]{8,}(?!\w)"),
        base_score=0.64,
        hint="device",
        priority=35,
    ),
    # ------------------------------------------------------------------ #
    # TRAVEL / ITINERARY (contextual PII)
    # ------------------------------------------------------------------ #
    HeuristicRule(
        id="travel_plan",
        pii_type="TRAVEL_PLAN",
        category="LOCATION_ACTIVITY",
        any_keywords=("itinerary", "flight", "train", "hotel booking", "check-in"),
        requires_digits=False,
        base_score=0.55,
        right_context=48,
        hint="location",
        priority=65,
    ),
)


__all__ = ["HeuristicRule", "HEURISTIC_RULES", "HEURISTIC_VERSION"]
