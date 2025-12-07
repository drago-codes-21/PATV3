"""Configurable regex pattern definitions for detectors."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class PatternDefinition:
    """Declarative pattern specification used by the regex detector."""

    id: str
    pii_type: str
    regex: str
    category: str | None = None
    confidence: float = 0.9
    flags: int = re.IGNORECASE
    group: int | None = None
    validator: str | None = None
    candidate: bool = False
    keywords: tuple[str, ...] = ()


# Negative/allowlist controls to suppress noisy matches.
SAFE_EMAIL_DOMAINS: set[str] = {
    "noreply.example.com",
    "noreply.example.co.uk",
    "support.example.com",
    "example.com",
    "localhost",
}

ALLOWED_LITERALS: set[str] = {
    "ORDER-12345678",
    "TICKET-00000000",
}

# Pattern IDs that should be dropped entirely if matched.
NEGATIVE_PATTERN_IDS: set[str] = set()

# Configurable pattern library.
PATTERN_DEFINITIONS: Sequence[PatternDefinition] = (
    PatternDefinition(
        id="sort_code_keyworded",
        pii_type="SORT_CODE",
        category="FINANCIAL",
        regex=r"\bsort\s*code\s*(?:[:\-]|is)?\s*(\d{2}[- ]?\d{2}[- ]?\d{2})\b",
        confidence=0.95,
        group=1,
        validator="sort_code",
    ),
    PatternDefinition(
        id="sort_code_compact_keyworded",
        pii_type="SORT_CODE",
        category="FINANCIAL",
        regex=r"\bsort\s*code\s*(?:[:\-]|is)?\s*(\d{6})\b",
        confidence=0.9,
        group=1,
        validator="sort_code",
    ),
    PatternDefinition(
        id="account_number_keyworded",
        pii_type="BANK_ACCOUNT",
        category="FINANCIAL",
        regex=r"\b(?:account|acct|a\/c)\s*(?:number|no\.?)?\s*[:\-]?\s*(\d{8})\b",
        confidence=0.92,
        group=1,
        validator="account_number",
    ),
    PatternDefinition(
        id="gb_iban",
        pii_type="IBAN",
        category="FINANCIAL",
        regex=r"\b(GB\d{2}[A-Z]{4}[A-Z0-9]{14})\b",
        confidence=0.95,
        group=1,
        validator="gb_iban",
    ),
    PatternDefinition(
        id="gb_iban_spaced",
        pii_type="IBAN",
        category="FINANCIAL",
        regex=r"\b(GB\d{2}\s?[A-Z]{4}(?:\s?\d{2}){7})\b",
        confidence=0.9,
        group=1,
        validator="gb_iban",
    ),
    PatternDefinition(
        id="card_number_grouped",
        pii_type="CARD_NUMBER",
        category="FINANCIAL",
        # Matches common groupings for 15-16 digit cards (e.g., Visa, MC, Amex)
        regex=r"\b(\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}|\d{4}[ -]?\d{6}[ -]?\d{5})\b",
        confidence=0.95,
        group=1,
        validator="card_luhn",
    ),
    PatternDefinition(
        id="card_number_contiguous",
        pii_type="CARD_NUMBER",
        category="FINANCIAL",
        regex=r"\b(\d{13,19})\b",
        confidence=0.8,  # Slightly lower confidence as it might be other long numbers
        group=1,
        validator="card_luhn",
    ),
    PatternDefinition(
        id="card_number_keyworded_generic",
        pii_type="CARD_NUMBER",
        category="FINANCIAL",
        regex=r"\b((?:\d[ -]?){12,18}\d)\b",  # Generic pattern for 13-19 digits with separators
        confidence=0.7,  # Requires keywords to reduce false positives
        group=1,
        validator="card_luhn",
        keywords=("card", "debit", "credit", "visa", "mastercard", "amex", "maestro", "pan", "ccv", "cc"),
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
    ),
    PatternDefinition(
        id="email_standard",
        pii_type="EMAIL",
        category="CONTACT",
        regex=r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b",
        confidence=0.9,
        group=1,
        validator="email_safe",
    ),
    PatternDefinition(
        id="postcode_uk",
        pii_type="POSTCODE",
        category="CONTACT",
        regex=r"\b([A-Z]{1,2}\d[A-Z\d]?\s?\d[ABD-HJLN-UW-Z]{2})\b",
        confidence=0.86,
        group=1,
        validator="postcode",
    ),
    PatternDefinition(
        id="phone_keyworded",
        pii_type="PHONE",
        category="CONTACT",
        regex=r"\b(\+?\d[\d\-\s()]{8,}\d)\b",
        confidence=0.82,
        group=1,
        validator="phone_len",
        keywords=("phone", "mobile", "number", "call"),
    ),
    PatternDefinition(
        id="address_uk_simple",
        pii_type="ADDRESS",
        category="CONTACT",
        regex=r"\b(\d{1,4}\s+[A-Za-z]{2,}(?:\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Close|Cl)))\b",
        confidence=0.8,
        group=1,
        keywords=("ship", "address", "deliver", "send", "to"),
    ),
    PatternDefinition(
        id="money_currency",
        pii_type="MONEY",
        category="FINANCIAL",
        regex=r"([£Ł$€L]\s?\d{1,6}(?:[.,]\d{2})?)\b",
        confidence=0.8,
        group=1,
        validator="money",
    ),
    PatternDefinition(
        id="customer_id_prefixed",
        pii_type="CUSTOMER_ID",
        category="FINANCIAL",
        regex=r"\b(CUST-?\d{4,12})\b",
        confidence=0.8,
        group=1,
    ),
    PatternDefinition(
        id="api_key_prefixed",
        pii_type="API_KEY",
        category="AUTHENTICATION",
        regex=r"\b(API-[A-Za-z0-9]{12,64})\b",
        confidence=0.88,
        group=1,
    ),
    PatternDefinition(
        id="credential_prefixed",
        pii_type="CREDENTIAL",
        category="AUTHENTICATION",
        regex=r"\b(cred-[A-Za-z0-9]{6,40})\b",
        confidence=0.86,
        group=1,
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
    ),
    PatternDefinition(
        id="password_keyworded",
        pii_type="PASSWORD",
        category="AUTHENTICATION",
        regex=r"\b(?:password|passcode|pwd)[:=]?\s*([A-Za-z0-9!@#$%^&*]{6,64})\b",
        confidence=0.82,
        group=1,
        validator="password",
    ),
    PatternDefinition(
        id="device_id_prefixed",
        pii_type="DEVICE_ID",
        category="ONLINE_DEVICE",
        regex=r"\b(dev-[a-f0-9]{10,32})\b",
        confidence=0.85,
        group=1,
    ),
    PatternDefinition(
        id="student_id_prefixed",
        pii_type="STUDENT_ID",
        category="EMPLOYMENT_EDUCATION",
        regex=r"\b(SID\d{3,12})\b",
        confidence=0.82,
        group=1,
    ),
    PatternDefinition(
        id="driving_license_alt",
        pii_type="DRIVING_LICENSE",
        category="GOVERNMENT_ID",
        regex=r"\bDL[-\s]?(\d{9,12})\b",
        confidence=0.8,
        group=1,
    ),
    PatternDefinition(
        id="date_simple",
        pii_type="DATE",
        category="CONTACT",
        regex=r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        confidence=0.7,
        group=1,
        validator="date",
        keywords=("date", "on", "due", "event"),
    ),
    PatternDefinition(
        id="nino",
        pii_type="NI_NUMBER",
        category="GOVERNMENT_ID",
        regex=r"\b([A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D])\b",
        confidence=0.9,
        group=1,
        validator="nino",
    ),
    PatternDefinition(
        id="nhs_number",
        pii_type="NHS_NUMBER",
        category="GOVERNMENT_ID",
        regex=r"\b(\d{3}\s?\d{3}\s?\d{4})\b",
        confidence=0.9,
        group=1,
        validator="nhs_number",
    ),
    PatternDefinition(
        id="passport_keyworded",
        pii_type="PASSPORT_NUMBER",
        category="GOVERNMENT_ID",
        regex=r"\b(\d{9})\b",
        confidence=0.82,
        group=1,
        validator="passport",
        keywords=("passport",),
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
    ),
    PatternDefinition(
        id="ip_v4",
        pii_type="IP_ADDRESS",
        category="ONLINE_DEVICE",
        regex=r"\b((?:\d{1,3}\.){3}\d{1,3})\b",
        confidence=0.9,
        group=1,
        validator="ipv4",
    ),
    PatternDefinition(
        id="ip_v6",
        pii_type="IPV6_ADDRESS",
        category="ONLINE_DEVICE",
        # Broad IPv6 matcher including compressed forms; validator ensures correctness.
        regex=r"\b([0-9a-f]{0,4}(?::[0-9a-f]{0,4}){2,7})\b",
        confidence=0.86,
        group=1,
        validator="ipv6",
        flags=re.IGNORECASE,
    ),
    PatternDefinition(
        id="mac_address",
        pii_type="MAC_ADDRESS",
        category="ONLINE_DEVICE",
        regex=r"\b([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})\b",
        confidence=0.88,
        group=1,
        validator="mac_address",
    ),
    PatternDefinition(
        id="url_basic",
        pii_type="URL",
        category="CONTACT",
        regex=r"\bhttps?://[^\s<>\"']+\b",
        confidence=0.75,
        group=0,
        validator="url",
    ),
    PatternDefinition(
        id="gps_coordinates",
        pii_type="GEO_COORDINATES",
        category="LOCATION_ACTIVITY",
        regex=r"\b(-?\d{1,2}\.\d{4,}),\s*(-?\d{1,3}\.\d{4,})\b",
        confidence=0.8,
        group=0,
        validator="gps",
    ),
)


def get_compiled_patterns(definitions: Iterable[PatternDefinition] | None = None) -> list[PatternDefinition]:
    """
    Return PatternDefinitions with compiled regex objects for efficiency.
    Note: This corrects the original implementation which did not compile regexes.
    """
    source_definitions = definitions or PATTERN_DEFINITIONS
    # The `re` module caches compiled regexes, but explicit compilation is a best practice
    # and allows us to store the compiled object directly in our definition.
    return [
        # Unpack the spec and overwrite the regex string with a compiled object.
        PatternDefinition(**{**spec.__dict__, "regex": re.compile(spec.regex, spec.flags)}) # type: ignore
        for spec in source_definitions
    ]


__all__: List[str] = [
    "PatternDefinition",
    "PATTERN_DEFINITIONS",
    "SAFE_EMAIL_DOMAINS",
    "ALLOWED_LITERALS",
    "NEGATIVE_PATTERN_IDS",
]
