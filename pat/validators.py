"""
Validator registry for the RegexDetector.

These functions perform structural or semantic checks on regex-captured spans.
All validators:
    - Must be deterministic
    - Must not raise exceptions
    - Must not modify global state
    - Must operate ONLY on match.group(1) unless otherwise documented
"""

from __future__ import annotations

import ipaddress
import re
from typing import Callable

from pat.config.patterns import SAFE_EMAIL_DOMAINS


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _extract_group(match: re.Match) -> str:
    """Safely return match.group(1) with defensive guards."""
    try:
        return match.group(1)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Luhn checksum (credit/debit cards)
# ---------------------------------------------------------------------------

def is_luhn_valid(match: re.Match) -> bool:
    num = re.sub(r"[ -]", "", _extract_group(match))
    if not num.isdigit() or len(num) < 12:
        return False

    digits = [int(d) for d in num]
    checksum = sum(digits[-1::-2]) + sum(sum(divmod(d * 2, 10)) for d in digits[-2::-2])
    return checksum % 10 == 0


# ---------------------------------------------------------------------------
# UK Sort Code
# ---------------------------------------------------------------------------

def is_sort_code_valid(match: re.Match) -> bool:
    code = re.sub(r"[- ]", "", _extract_group(match))
    # Additional guard: prevent nonsense like 000000
    return len(code) == 6 and code.isdigit() and not all(c == "0" for c in code)


# ---------------------------------------------------------------------------
# GB IBAN
# ---------------------------------------------------------------------------

def is_gb_iban_valid(match: re.Match) -> bool:
    raw = re.sub(r"\s", "", _extract_group(match)).upper()
    if len(raw) != 22 or not raw.startswith("GB"):
        return False

    # Rearrange for mod97: move first 4 chars to the end
    rearranged = raw[4:] + raw[:4]

    # Convert letters to digits (A=10, ..., Z=35)
    converted = []
    for ch in rearranged:
        if ch.isdigit():
            converted.append(ch)
        elif "A" <= ch <= "Z":
            converted.append(str(ord(ch) - 55))
        else:
            return False

    try:
        return int("".join(converted)) % 97 == 1
    except Exception:
        return False


# ---------------------------------------------------------------------------
# UK National Insurance (NINO)
# ---------------------------------------------------------------------------

def is_nino_valid(match: re.Match) -> bool:
    nino = _extract_group(match).upper()
    if len(nino) != 9:
        return False
    # Accept "QQ" prefix for synthetic test numbers
    return bool(re.fullmatch(r"[A-Z]{2}\d{6}[A-D]", nino))


# ---------------------------------------------------------------------------
# NHS number (mod11 checksum)
# ---------------------------------------------------------------------------

def is_nhs_number_valid(match: re.Match) -> bool:
    num = re.sub(r"\s", "", _extract_group(match))
    if len(num) != 10 or not num.isdigit():
        return False

    digits = [int(d) for d in num]
    weights = list(range(10, 1, -1))
    total = sum(d * w for d, w in zip(digits[:9], weights))
    remainder = total % 11

    check_digit = 11 - remainder
    if check_digit == 11:
        check_digit = 0
    if check_digit == 10:
        return False

    return check_digit == digits[9]


# ---------------------------------------------------------------------------
# Email domain safety check
# ---------------------------------------------------------------------------

def is_email_safe(match: re.Match) -> bool:
    email = _extract_group(match)
    if "@" not in email:
        return False
    try:
        domain = email.split("@", 1)[1].lower()
        return domain not in SAFE_EMAIL_DOMAINS
    except Exception:
        return False


# ---------------------------------------------------------------------------
# IP validation
# ---------------------------------------------------------------------------

def is_ipv4_valid(match: re.Match) -> bool:
    try:
        ipaddress.IPv4Address(_extract_group(match))
        return True
    except Exception:
        return False


def is_ipv6_valid(match: re.Match) -> bool:
    try:
        ipaddress.IPv6Address(_extract_group(match))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# PIN validation
# ---------------------------------------------------------------------------

def is_pin_valid(match: re.Match) -> bool:
    pin = _extract_group(match)
    return pin.isdigit() and 4 <= len(pin) <= 6 and not (pin == pin[0] * len(pin))


# ---------------------------------------------------------------------------
# Password heuristic (very conservative)
# ---------------------------------------------------------------------------

def is_password_like(match: re.Match) -> bool:
    pwd = _extract_group(match)
    # Reject trivial passwords
    if len(pwd) < 6:
        return False
    if pwd.lower() in {"password", "passcode", "pwd"}:
        return False
    if pwd.isdigit():
        return False
    return True


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------

def is_date_valid(match: re.Match) -> bool:
    text = _extract_group(match)
    parts = re.split(r"[/-]", text)

    if len(parts) != 3:
        return False

    try:
        d, m, y = map(int, parts)
    except Exception:
        return False

    if not (1 <= d <= 31 and 1 <= m <= 12 and 0 <= y <= 9999):
        return False

    return True


# ---------------------------------------------------------------------------
# Money (simple guard)
# ---------------------------------------------------------------------------

def is_money_valid(match: re.Match) -> bool:
    raw = _extract_group(match)
    digits = re.sub(r"\D", "", raw)
    return 1 <= len(digits) <= 8


# ---------------------------------------------------------------------------
# Phone length validator
# ---------------------------------------------------------------------------

def is_phone_len_valid(match: re.Match) -> bool:
    digits = re.sub(r"\D", "", _extract_group(match))
    return 10 <= len(digits) <= 15


# ---------------------------------------------------------------------------
# Roll number validator
# ---------------------------------------------------------------------------

def is_roll_number_valid(match: re.Match) -> bool:
    text = _extract_group(match)
    if not (7 <= len(text) <= 20):
        return False
    return any(c.isdigit() for c in text) and any(c.isalpha() for c in text)


# ---------------------------------------------------------------------------
# GPS sanity check
# ---------------------------------------------------------------------------

def is_gps_valid(match: re.Match) -> bool:
    try:
        lat_str, lon_str = match.group(1), match.group(2)
        lat = float(lat_str)
        lon = float(lon_str)
        return -90 <= lat <= 90 and -180 <= lon <= 180
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VALIDATORS: dict[str, Callable[[re.Match], bool]] = {
    "card_luhn": is_luhn_valid,
    "sort_code": is_sort_code_valid,
    "account_number": lambda m: True,  # No checksum format for UK 8-digit accounts
    "gb_iban": is_gb_iban_valid,
    "roll_number": is_roll_number_valid,
    "email_safe": is_email_safe,
    "postcode": lambda m: True,  # Regex already enforces format
    "phone_len": is_phone_len_valid,
    "nino": is_nino_valid,
    "nhs_number": is_nhs_number_valid,
    "passport": lambda m: True,  # keyword-gated upstream
    "driving_license": lambda m: True,
    "money": is_money_valid,
    "pin": is_pin_valid,
    "password": is_password_like,
    "date": is_date_valid,
    "ipv4": is_ipv4_valid,
    "ipv6": is_ipv6_valid,
    "mac_address": lambda m: True,
    "url": lambda m: True,
    "gps": is_gps_valid,
}
