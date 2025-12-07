"""Masking utilities for policy engine."""

from __future__ import annotations

import re
from typing import Any, Callable


def mask_full(text: str, char: str = "*") -> str:
    return char * len(text)


def mask_preserve_last_n(text: str, char: str, n: int) -> str:
    n = max(0, n)
    if n >= len(text):
        return text
    masked = mask_full(text[:-n], char) + text[-n:]
    return masked


def mask_email(text: str, char: str = "*", keep_domain: bool = True) -> str:
    if "@" not in text:
        return mask_full(text, char)
    local, domain = text.split("@", 1)
    if not local:
        return mask_full(text, char)
    masked_local = local[0] + (char * max(0, len(local) - 1))
    if keep_domain:
        return f"{masked_local}@{domain}"
    return f"{masked_local}@{mask_full(domain, char)}"


def mask_preserve_format(text: str, char: str = "*") -> str:
    """Mask digits/letters, preserve punctuation/spacing."""
    out = []
    for ch in text:
        if ch.isdigit() or ch.isalpha():
            out.append(char)
        else:
            out.append(ch)
    return "".join(out)


def mask_semantic(text: str, replacement: str = "[REDACTED]") -> str:
    return replacement


def mask_credential(text: str, char: str = "#") -> str:
    # Preserve length to avoid structural leaks.
    return mask_full(text, char)


def mask_numeric_token(text: str, char: str = "*") -> str:
    return re.sub(r"\d", char, text)


def mask_placeholder(_: str, placeholder: str = "[REDACTED]") -> str:
    return placeholder


STRATEGY_DISPATCH: dict[str, Callable[..., str]] = {
    "full": mask_full,
    "preserve_lastN": mask_preserve_last_n,
    "preserve_last_n": mask_preserve_last_n,
    "partial_email": mask_email,
    "preserve_format": mask_preserve_format,
    "semantic_mask": mask_semantic,
    "mask_credential": mask_credential,
    "numeric_token": mask_numeric_token,
    "placeholder": mask_placeholder,
}


def apply_mask(mask_style: str, text: str, mask_args: dict[str, Any] | None = None) -> str:
    """Dispatch masking strategy with graceful fallback to full masking."""

    args = mask_args or {}
    style = (mask_style or "full").lower()
    try:
        if style in {"preserve_lastn", "preserve_last_n"}:
            return mask_preserve_last_n(text, args.get("mask_char", "*"), int(args.get("last_n", args.get("visible_count", 4))))
        if style == "partial_email":
            return mask_email(text, args.get("mask_char", "*"), bool(args.get("keep_domain", True)))
        if style == "preserve_format":
            return mask_preserve_format(text, args.get("mask_char", "*"))
        if style == "semantic_mask":
            return mask_semantic(text, args.get("replacement", "[REDACTED]"))
        if style == "mask_credential":
            return mask_credential(text, args.get("mask_char", "#"))
        if style == "numeric_token":
            return mask_numeric_token(text, args.get("mask_char", "*"))
        if style == "placeholder":
            return mask_placeholder(text, args.get("placeholder", "[REDACTED]"))
        if style == "full":
            return mask_full(text, args.get("mask_char", "*"))
    except Exception:
        # Fall through to full masking below on any error.
        ...
    return mask_full(text, args.get("mask_char", "*"))
