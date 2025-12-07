"""Configuration for heuristic, context-aware PII rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern, Sequence, Tuple


@dataclass(frozen=True)
class HeuristicRule:
    """Declarative heuristic rule."""

    id: str
    pii_type: str
    any_keywords: Tuple[str, ...]
    all_keywords: Tuple[str, ...] = ()
    context_window: int = 64
    requires_digits: bool = False
    number_pattern: Pattern[str] = re.compile(r"\d[\d\s-]{2,}\d")
    base_score: float = 0.6
    allow_partial_window: bool = True
    hint: str | None = None


HEURISTIC_RULES: Sequence[HeuristicRule] = (
    HeuristicRule(
        id="sort_code",
        pii_type="SORT_CODE",
        any_keywords=("sort code", "sort-code"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)(\d{2}[-\s]?\d{2}[-\s]?\d{2})(?!\d)"),
        base_score=0.75,
        hint="financial",
    ),
    HeuristicRule(
        id="account_number",
        pii_type="BANK_ACCOUNT",
        any_keywords=("account number", "account no", "acct no", "iban"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)(\d{8,18})(?!\d)"),
        base_score=0.72,
        hint="financial",
    ),
    HeuristicRule(
        id="card_number",
        pii_type="CARD_NUMBER",
        any_keywords=("card number", "debit card", "credit card", "pan"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
        base_score=0.78,
        hint="financial",
    ),
    HeuristicRule(
        id="swift_routing",
        pii_type="BANK_ACCOUNT",
        any_keywords=("swift", "routing", "sort code", "ifsc"),
        requires_digits=True,
        number_pattern=re.compile(r"\b[A-Z]{4}\w{7}\b|\b\d{6,12}\b", flags=re.IGNORECASE),
        base_score=0.62,
        hint="financial",
    ),
    HeuristicRule(
        id="credential_password",
        pii_type="CREDENTIAL",
        any_keywords=("password", "passcode", "pass code", "pwd", "passphrase"),
        requires_digits=False,
        base_score=0.85,
        hint="credential",
    ),
    HeuristicRule(
        id="credential_token",
        pii_type="CREDENTIAL",
        any_keywords=("secret", "token", "api key", "api-key", "bearer token"),
        requires_digits=False,
        base_score=0.83,
        hint="credential",
    ),
    HeuristicRule(
        id="credential_otp",
        pii_type="CREDENTIAL",
        any_keywords=("otp", "one-time passcode", "one time password", "verification code"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)\d{4,8}(?!\d)"),
        base_score=0.8,
        hint="credential",
    ),
    HeuristicRule(
        id="credential_pin",
        pii_type="CREDENTIAL",
        any_keywords=("pin", "security number"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)\d{4,6}(?!\d)"),
        base_score=0.76,
        hint="credential",
    ),
    HeuristicRule(
        id="gov_identity",
        pii_type="GOV_ID",
        any_keywords=("ssn", "social security", "passport", "driver", "license", "aadhaar", "pan"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)(\d{6,12})(?!\d)"),
        base_score=0.7,
        hint="identity",
    ),
    HeuristicRule(
        id="contact_phone",
        pii_type="PHONE",
        any_keywords=("phone", "mobile", "cell", "whatsapp"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\d)(\+?\d[\d\-\s()]{6,}\d)(?!\d)"),
        base_score=0.68,
        hint="contact",
    ),
    HeuristicRule(
        id="contact_address",
        pii_type="ADDRESS",
        any_keywords=("home address", "residence", "shipping address", "billing address"),
        requires_digits=False,
        base_score=0.65,
        hint="location",
    ),
    HeuristicRule(
        id="medical_context",
        pii_type="MEDICAL_INFO",
        any_keywords=("diagnosis", "diagnosed", "treatment", "symptom", "medical condition", "allergy"),
        requires_digits=False,
        base_score=0.7,
        hint="health",
    ),
    HeuristicRule(
        id="medication_context",
        pii_type="MEDICATION",
        any_keywords=("medication", "medicine", "prescription", "rx", "tablet", "pill"),
        requires_digits=False,
        base_score=0.65,
        hint="health",
    ),
    HeuristicRule(
        id="hospital_context",
        pii_type="HOSPITAL",
        any_keywords=("hospital", "clinic", "nhs trust"),
        requires_digits=False,
        base_score=0.62,
        hint="health",
    ),
    HeuristicRule(
        id="employment_employer",
        pii_type="EMPLOYER_NAME",
        any_keywords=("employer", "company", "works at", "employed by"),
        requires_digits=False,
        base_score=0.6,
        hint="employment",
    ),
    HeuristicRule(
        id="employment_id",
        pii_type="EMPLOYMENT_ID",
        any_keywords=("employee id", "staff id", "payroll id"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\w)[A-Z]?\d{4,10}(?!\w)", flags=re.IGNORECASE),
        base_score=0.65,
        hint="employment",
    ),
    HeuristicRule(
        id="job_title",
        pii_type="JOB_TITLE",
        any_keywords=("job title", "role", "position"),
        requires_digits=False,
        base_score=0.55,
        hint="employment",
    ),
    HeuristicRule(
        id="biometric_context",
        pii_type="BIOMETRIC",
        any_keywords=("fingerprint", "face id", "voiceprint", "iris scan", "retina"),
        requires_digits=False,
        base_score=0.72,
        hint="biometric",
    ),
    HeuristicRule(
        id="device_id",
        pii_type="DEVICE_ID",
        any_keywords=("device id", "imei", "udid", "android id"),
        requires_digits=True,
        number_pattern=re.compile(r"(?<!\w)[A-Fa-f0-9]{8,}(?!\w)"),
        base_score=0.64,
        hint="device",
    ),
    HeuristicRule(
        id="travel_plan",
        pii_type="TRAVEL_PLAN",
        any_keywords=("itinerary", "flight", "train", "hotel booking", "check-in"),
        requires_digits=False,
        base_score=0.55,
        hint="location",
    ),
)


__all__ = ["HeuristicRule", "HEURISTIC_RULES"]
