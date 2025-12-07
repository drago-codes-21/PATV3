"""
Static regex pattern definitions for PAT.

This module provides the authoritative list of PatternDefinition objects
used by the RegexDetector. Patterns are:

- precision-oriented (avoid over-masking)
- validator-backed where possible
- deterministically ordered (priority → id)
- compiled once for performance
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, List, Mapping, Sequence


# ---------------------------------------------------------------------------
# Versioning: Used for tooling + debugging
# ---------------------------------------------------------------------------

PATTERN_VERSION = "1.2.0"  # bumped for safety improvements + normalization


# ---------------------------------------------------------------------------
# Pattern schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatternDefinition:
    """
    Declarative regex pattern specification.

    Fields:
        id: Stable unique identifier (do not change; impacts NEGATIVE_PATTERN_IDS)
        pii_type: Output type assigned to matches
        regex: Raw pattern string OR compiled pattern (after compilation)
        group: group index or name for extracting the PII payload
        validator: Name of validator in pat.validators (optional)
        keywords: Additional heuristics used by detectors (tuple)
        candidate: Reserved for ML hybrid detectors
        priority: Lower number → earlier evaluation in RegexDetector
    """

    id: str
    pii_type: str
    regex: str | re.Pattern[str]
    category: str | None = None
    confidence: float = 0.9
    flags: int = re.IGNORECASE
    group: int | str | None = 0
    validator: str | None = None
    candidate: bool = False
    keywords: tuple[str, ...] = ()
    priority: int = 100


# ---------------------------------------------------------------------------
# Control lists
# ---------------------------------------------------------------------------

SAFE_EMAIL_DOMAINS: set[str] = {
    "noreply.example.com",
    "noreply.example.co.uk",
    "support.example.com",
    "example.com",
    "localhost",
}

# Exact or contextual patterns that should be skipped.
# Considered "always safe" when matched directly.
ALLOWED_LITERALS: set[str] = {
    "ORDER-12345678",
    "TICKET-00000000",
}

# Optional runtime suppressions
NEGATIVE_PATTERN_IDS: set[str] = set()


# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------

PATTERN_DEFINITIONS: Sequence[PatternDefinition] = (
    # FINANCIAL
    PatternDefinition(
        id="sort_code_keyworded",
        pii_type="SORT_CODE",
        category="FINANCIAL",
        regex=r"\bsort\s*code\s*(?:[:\-]|is)?\s*(\d{2}[- ]?\d{2}[- ]?\d{2})\b",
        confidence=0.95,
        group=1,
        validator="sort_code",
        priority=10,
    ),
    PatternDefinition(
        id="sort_code_compact_keyworded",
        pii_type="SORT_CODE",
        category="FINANCIAL",
        regex=r"\bsort\s*code\s*(?:[:\-]|is)?\s*(\d{6})\b",
        confidence=0.9,
        group=1,
        validator="sort_code",
        priority=15,
    ),
    PatternDefinition(
        id="account_number_keyworded",
        pii_type="BANK_ACCOUNT",
        category="FINANCIAL",
        regex=r"\b(?:account|acct|a\/c)\s*(?:number|no\.?)?\s*[:\-]?\s*(\d{8})\b",
        confidence=0.92,
        group=1,
        validator="account_number",
        priority=10,
    ),
    PatternDefinition(
        id="gb_iban",
        pii_type="IBAN",
        category="FINANCIAL",
        regex=r"\b(GB\d{2}[A-Z]{4}[A-Z0-9]{14})\b",
        confidence=0.95,
        group=1,
        validator="gb_iban",
        priority=5,
    ),
    PatternDefinition(
        id="gb_iban_spaced",
        pii_type="IBAN",
        category="FINANCIAL",
        regex=r"\b(GB\d{2}\s?[A-Z]{4}(?:\s?\d{2}){7})\b",
        confidence=0.9,
        group=1,
        validator="gb_iban",
        priority=6,
    ),
    PatternDefinition(
        id="card_number_grouped",
        pii_type="CARD_NUMBER",
        category="FINANCIAL",
        regex=r"\b(\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}|\d{4}[ -]?\d{6}[ -]?\d{5})\b",
        confidence=0.95,
        group=1,
        validator=None,
        priority=5,
    ),
    PatternDefinition(
        id="card_number_contiguous",
        pii_type="CARD_NUMBER",
        category="FINANCIAL",
        regex=r"\b([2-6]\d{12,18})\b",
        confidence=0.75,
        group=1,
        validator="card_luhn",
        keywords=(
            "card", "debit", "credit", "visa", "mastercard", "amex", "maestro", "pan", "cc"
        ),
        priority=20,
    ),
    PatternDefinition(
        id="card_number_keyworded_generic",
        pii_type="CARD_NUMBER",
        category="FINANCIAL",
        regex=r"\b((?:\d[ -]?){12,18}\d)\b",
        confidence=0.7,
        group=1,
        validator="card_luhn",
        keywords=(
            "card", "debit", "credit", "visa", "mastercard", "amex", "maestro", "pan", "cc"
        ),
        priority=25,
    ),
    PatternDefinition(
        id="building_society_roll",
        pii_type="CUSTOMER_ID",
        category="FINANCIAL",
        regex=r"\broll\s*(?:number|no)?[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-\/\.]{5,18}[A-Za-z0-9])\b",
        confidence=0.9,
        group=1,
        validator="roll_number",
        keywords=("roll",),
        priority=30,
    ),
    PatternDefinition(
        id="money_currency",
        pii_type="MONEY",
        category="FINANCIAL",
        regex=r"([£$€]\s?\d{1,6}(?:[.,]\d{2})?)\b",
        confidence=0.8,
        group=1,
        validator="money",
        priority=40,
    ),
    PatternDefinition(
        id="customer_id_prefixed",
        pii_type="CUSTOMER_ID",
        category="FINANCIAL",
        regex=r"\b(CUST-?\d{4,12})\b",
        confidence=0.8,
        group=1,
        priority=50,
    ),
    # AUTHENTICATION
    PatternDefinition(
        id="api_key_prefixed",
        pii_type="API_KEY",
        category="AUTHENTICATION",
        regex=r"\b(API-[A-Za-z0-9]{12,64})\b",
        confidence=0.88,
        group=1,
        priority=20,
    ),
    PatternDefinition(
        id="credential_prefixed",
        pii_type="CREDENTIAL",
        category="AUTHENTICATION",
        regex=r"\b(cred-[A-Za-z0-9]{6,40})\b",
        confidence=0.86,
        group=1,
        priority=25,
    ),
    PatternDefinition(
        id="pin_keyworded",
        pii_type="PIN",
        category="AUTHENTICATION",
        regex=r"\bpin[:=]?\s*(\d{4,6})\b",
        confidence=0.8,
        group=1,
        validator="pin",
        keywords=("pin",),
        priority=30,
    ),
    PatternDefinition(
        id="password_keyworded",
        pii_type="PASSWORD",
        category="AUTHENTICATION",
        regex=r"\b(?:password|passcode|pwd)[:=]?\s*([A-Za-z0-9!@#$%^&*]{6,64})\b",
        confidence=0.82,
        group=1,
        validator="password",
        priority=30,
    ),
    # CONTACT
    PatternDefinition(
        id="email_standard",
        pii_type="EMAIL",
        category="CONTACT",
        regex=r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24})\b",
        confidence=0.9,
        group=1,
        validator="email_safe",
        priority=5,
    ),
    PatternDefinition(
        id="postcode_uk",
        pii_type="POSTCODE",
        category="CONTACT",
        regex=r"\b([A-Z]{1,2}\d[A-Z\d]?\s?\d[ABD-HJLN-UW-Z]{2})\b",
        confidence=0.86,
        group=1,
        validator="postcode",
        priority=20,
    ),
    PatternDefinition(
        id="phone_keyworded",
        pii_type="PHONE",
        category="CONTACT",
        regex=r"\b(\+?\d[\d\-\s()]{8,}\d)\b",
        confidence=0.82,
        group=1,
        validator="phone_len",
        keywords=("phone", "mobile", "call", "tel", "telephone"),
        priority=25,
    ),
    PatternDefinition(
        id="address_uk_simple",
        pii_type="ADDRESS",
        category="CONTACT",
        regex=(
            r"\b(\d{1,4}\s+[A-Za-z]{2,}"
            r"(?:\s+(?:Street|St\.?|Road|Rd\.?|Avenue|Ave|Lane|Ln|Close|Cl)))\b"
        ),
        confidence=0.8,
        group=1,
        keywords=("ship", "address", "deliver", "send", "to", "billing"),
        priority=40,
    ),
    PatternDefinition(
        id="url_basic",
        pii_type="URL",
        category="CONTACT",
        regex=r"\bhttps?://[^\s<>\"']+\b",
        confidence=0.75,
        group=0,
        validator="url",
        priority=30,
    ),
    # DATES
    PatternDefinition(
        id="date_simple",
        pii_type="DATE",
        category="CONTACT",
        regex=r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        confidence=0.7,
        group=1,
        validator="date",
        keywords=("date", "on", "due", "event", "dob", "birth"),
        priority=50,
    ),
    # GOVERNMENT
    PatternDefinition(
        id="nino",
        pii_type="NI_NUMBER",
        category="GOVERNMENT_ID",
        regex=r"\b([A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D])\b",
        confidence=0.9,
        group=1,
        validator="nino",
        priority=5,
    ),
    PatternDefinition(
        id="nhs_number",
        pii_type="NHS_NUMBER",
        category="GOVERNMENT_ID",
        regex=r"\b(\d{3}\s?\d{3}\s?\d{4})\b",
        confidence=0.9,
        group=1,
        validator="nhs_number",
        priority=5,
    ),
    PatternDefinition(
        id="passport_keyworded",
        pii_type="PASSPORT_NUMBER",
        category="GOVERNMENT_ID",
        regex=r"\b(\d{9})\b",
        confidence=0.82,
        group=1,
        validator="passport",
        keywords=("passport", "pp#", "passportno", "passport_number"),
        priority=30,
    ),
    PatternDefinition(
        id="driving_license_keyworded",
        pii_type="DRIVING_LICENSE",
        category="GOVERNMENT_ID",
        regex=r"\b([A-Z]{5}\d{6}[A-Z]{2}\d{2})\b",
        confidence=0.82,
        group=1,
        validator="driving_license",
        keywords=("driver", "licence", "license"),
        priority=20,
    ),
    PatternDefinition(
        id="driving_license_alt",
        pii_type="DRIVING_LICENSE",
        category="GOVERNMENT_ID",
        regex=r"\bDL[-\s]?(\d{9,12})\b",
        confidence=0.8,
        group=1,
        priority=30,
    ),
    # ONLINE
    PatternDefinition(
        id="ip_v4",
        pii_type="IP_ADDRESS",
        category="ONLINE_DEVICE",
        regex=r"\b((?:\d{1,3}\.){3}\d{1,3})\b",
        confidence=0.9,
        group=1,
        validator="ipv4",
        priority=10,
    ),
    PatternDefinition(
        id="ip_v6",
        pii_type="IPV6_ADDRESS",
        category="ONLINE_DEVICE",
        regex=r"\b([0-9a-f]{0,4}(?::[0-9a-f]{0,4}){2,7})\b",
        confidence=0.86,
        group=1,
        validator="ipv6",
        flags=re.IGNORECASE,
        priority=15,
    ),
    PatternDefinition(
        id="mac_address",
        pii_type="MAC_ADDRESS",
        category="ONLINE_DEVICE",
        regex=r"\b([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})\b",
        confidence=0.88,
        group=1,
        validator="mac_address",
        priority=20,
    ),
    PatternDefinition(
        id="device_id_prefixed",
        pii_type="DEVICE_ID",
        category="ONLINE_DEVICE",
        regex=r"\b(dev-[a-f0-9]{10,32})\b",
        confidence=0.85,
        group=1,
        priority=30,
    ),
    # EDUCATION
    PatternDefinition(
        id="student_id_prefixed",
        pii_type="STUDENT_ID",
        category="EMPLOYMENT_EDUCATION",
        regex=r"\b(SID\d{3,12})\b",
        confidence=0.82,
        group=1,
        priority=40,
    ),
    # LOCATION
    PatternDefinition(
        id="gps_coordinates",
        pii_type="GEO_COORDINATES",
        category="LOCATION_ACTIVITY",
        regex=r"\b(-?\d{1,2}\.\d{4,}),\s*(-?\d{1,3}\.\d{4,})\b",
        confidence=0.8,
        group=0,
        validator="gps",
        priority=30,
    ),
)


# ---------------------------------------------------------------------------
# Index for lookup by tools / evaluators
# ---------------------------------------------------------------------------

PATTERN_INDEX: Mapping[str, PatternDefinition] = {p.id: p for p in PATTERN_DEFINITIONS}


# ---------------------------------------------------------------------------
# Compilation helper
# ---------------------------------------------------------------------------

def get_compiled_patterns(
    definitions: Iterable[PatternDefinition] | None = None,
) -> list[PatternDefinition]:
    """
    Compile regex patterns once, return new PatternDefinition objects with
    compiled `regex` fields.

    IMPORTANT:
        - Ordering is preserved from PATTERN_DEFINITIONS.
        - Compilation does NOT modify global definitions.
    """
    source = list(definitions or PATTERN_DEFINITIONS)
    compiled: list[PatternDefinition] = []

    for spec in source:
        if isinstance(spec.regex, re.Pattern):
            compiled.append(spec)
            continue

        try:
            compiled_regex = re.compile(spec.regex, spec.flags)
        except re.error as exc:
            raise RuntimeError(
                f"Failed to compile regex for pattern '{spec.id}': {exc}"
            )

        compiled.append(
            replace(spec, regex=compiled_regex)
        )

    return compiled


__all__ = [
    "PatternDefinition",
    "PATTERN_VERSION",
    "PATTERN_DEFINITIONS",
    "PATTERN_INDEX",
    "SAFE_EMAIL_DOMAINS",
    "ALLOWED_LITERALS",
    "NEGATIVE_PATTERN_IDS",
    "get_compiled_patterns",
]
