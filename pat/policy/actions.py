"""Redaction helper utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class RedactionOperation:
    start: int
    end: int
    replacement: str


def mask_span(text: str, start: int, end: int, mask_char: str = "*") -> str:
    """Mask a portion of text with the supplied character."""

    replacement = mask_char * max(0, end - start)
    return text[:start] + replacement + text[end:]


def tokenize_span(text: str, start: int, end: int, token: str) -> str:
    """Replace a span with a token literal."""

    return text[:start] + token + text[end:]


def apply_operations(text: str, operations: Iterable[RedactionOperation]) -> str:
    """Apply span replacements in reverse order to avoid index drift."""

    updated = text
    for op in sorted(operations, key=lambda item: item.start, reverse=True):
        updated = updated[: op.start] + op.replacement + updated[op.end :]
    return updated

