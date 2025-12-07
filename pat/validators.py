"""A registry of validator functions for the regex detector."""

from __future__ import annotations

import ipaddress
import re
from typing import Callable

from pat.config.patterns import SAFE_EMAIL_DOMAINS


def is_luhn_valid(match: re.Match) -> bool:
    """Check if a number is valid using the Luhn algorithm."""
    card_number = re.sub(r"[ -]", "", match.group(1))
    if not card_number.isdigit():
        return False
    digits = [int(d) for d in card_number]
    checksum = sum(digits[-1::-2]) + sum(
        [sum(divmod(d * 2, 10)) for d in digits[-2::-2]]
    )
    return checksum % 10 == 0


def is_sort_code_valid(match: re.Match) -> bool:
    """Basic structural validation for UK sort codes."""
    code = re.sub(r"[- ]", "", match.group(1))
    return len(code) == 6 and code.isdigit()


def is_gb_iban_valid(match: re.Match) -> bool:
    """Validate a GB IBAN using the mod-97 checksum."""
    iban = re.sub(r"\s", "", match.group(1)).upper()
    if len(iban) != 22 or not iban.startswith("GB"):
        return False
    rearranged = iban[4:] + iban[:4]
    # Convert letters to numbers (A=10, B=11, ..., Z=35)
    converted = ""
    for ch in rearranged:
        if ch.isdigit():
            converted += ch
        else:
            converted += str(ord(ch) - 55)
    try:
        return int(converted) % 97 == 1
    except ValueError:
        return False


def is_nino_valid(match: re.Match) -> bool:
    """Validate the structure of a UK National Insurance Number."""
    nino = match.group(1).upper()
    # For production we allow synthetic prefixes (e.g., QQ) often used in test data.
    if len(nino) != 9:
        return False
    return bool(re.match(r"^[A-Z]{2}\d{6}[A-D]$", nino))


def is_nhs_number_valid(match: re.Match) -> bool:
    """Validate an NHS number using the mod-11 checksum."""
    nhs_number = re.sub(r"\s", "", match.group(1))
    if len(nhs_number) != 10 or not nhs_number.isdigit():
        return False
    digits = [int(d) for d in nhs_number]
    weights = list(range(10, 1, -1))  # 10 to 2
    total = sum(d * w for d, w in zip(digits[:9], weights))
    remainder = total % 11
    check_digit = 11 - remainder
    if check_digit == 11:
        check_digit = 0
    if check_digit == 10:
        return False
    return check_digit == digits[9]


def is_email_safe(match: re.Match) -> bool:
    """Check if an email domain is in the safe list."""
    email = match.group(1)
    domain = email.split("@")[1]
    return domain.lower() not in SAFE_EMAIL_DOMAINS


def is_ipv4_valid(match: re.Match) -> bool:
    """Validate an IPv4 address."""
    try:
        ipaddress.ip_address(match.group(1))
        return True
    except ValueError:
        return False


def is_ipv6_valid(match: re.Match) -> bool:
    """Validate an IPv6 address."""
    try:
        ipaddress.ip_address(match.group(1))
        return True
    except ValueError:
        return False


def always_true(match: re.Match) -> bool:
    """A default validator that always passes."""
    return True


def is_money_valid(match: re.Match) -> bool:
    """Basic currency sanity: numeric portion must be reasonable length."""
    value = re.sub(r"[^\d.,]", "", match.group(1))
    digits = re.sub(r"\D", "", value)
    return 1 <= len(digits) <= 8


def is_pin_valid(match: re.Match) -> bool:
    pin = match.group(1)
    return pin.isdigit() and 4 <= len(pin) <= 6


def is_password_like(match: re.Match) -> bool:
    pwd = match.group(1)
    return len(pwd) >= 6


def is_date_valid(match: re.Match) -> bool:
    """Check dd/mm/yyyy style dates are plausible."""
    parts = re.split(r"[/-]", match.group(1))
    if len(parts) != 3:
        return False
    try:
        d, m, y = map(int, parts)
        if y < 0 or m < 1 or m > 12 or d < 1 or d > 31:
            return False
    except ValueError:
        return False
    return True


# A central registry mapping validator names to functions.
VALIDATORS: dict[str, Callable[[re.Match], bool]] = {
    "card_luhn": is_luhn_valid,
    "sort_code": is_sort_code_valid,
    "account_number": always_true,  # No standard checksum
    "gb_iban": is_gb_iban_valid,
    "roll_number": lambda m: (any(c.isdigit() for c in m.group(1)) and any(c.isalpha() for c in m.group(1)) and 7 <= len(m.group(1)) <= 20),
    "email_safe": is_email_safe,
    "postcode": always_true,  # Regex is quite specific
    "phone_len": lambda m: 10 <= len(re.sub(r"\D", "", m.group(1))) <= 15,
    "nino": is_nino_valid,
    "nhs_number": is_nhs_number_valid,
    "passport": always_true,
    "driving_license": always_true,
    "money": is_money_valid,
    "pin": is_pin_valid,
    "password": is_password_like,
    "date": is_date_valid,
    "ipv4": is_ipv4_valid,
    "ipv6": is_ipv6_valid,
    "mac_address": always_true,
    "url": always_true,
    "gps": always_true,
}
