"""Text processing helper functions used across detectors and severity."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

SENTENCE_DELIMITERS = ".!?\n"
TRIM_CHARS = " \t\r\n\"'`“”‘’[](){}<>….,;:!?|/"


def normalize_text(text: str) -> str:
    """Apply lightweight, reversible normalisation suitable for detector input."""

    if not text:
        return ""
    # NFKC pulls visually similar unicode into a consistent representation so
    # downstream regex/token alignment is stable across pasted email dumps.
    return unicodedata.normalize("NFKC", text)


def compute_sentence_boundaries(
    text: str,
    *,
    delimiters: Iterable[str] = SENTENCE_DELIMITERS,
    max_length: int = 500,
) -> list[tuple[int, int]]:
    """Split text into sentence-like windows and return index boundaries.

    The splitter is intentionally simple but guards against extremely long
    segments by forcing a break on whitespace once `max_length` characters are
    exceeded. This keeps chunks within the mpnet token budget to avoid
    truncation while preserving character offsets.
    """

    if not text:
        return [(0, 0)]

    delimiter_set = set(delimiters)
    boundaries: list[tuple[int, int]] = []
    start = 0

    for idx, char in enumerate(text):
        hit_delimiter = char in delimiter_set
        over_budget = (idx - start) >= max_length and char.isspace()
        if hit_delimiter or over_budget:
            end = idx + 1 if hit_delimiter else idx
            if end > start:
                boundaries.append((start, end))
            start = end

    if start < len(text):
        boundaries.append((start, len(text)))

    if not boundaries:
        boundaries.append((0, len(text)))
    return boundaries


def trim_span(text: str, start: int, end: int) -> tuple[int, int, str]:
    """Trim whitespace/punctuation from a span and return adjusted boundaries."""

    start = max(0, start)
    end = min(len(text), end)
    while start < end and text[start] in TRIM_CHARS:
        start += 1
    while end > start and text[end - 1] in TRIM_CHARS:
        end -= 1
    return start, end, text[start:end]


def split_with_token_budget(
    text: str,
    offsets: Sequence[tuple[int, int]],
    *,
    max_tokens: int,
    stride: int = 32,
) -> list[tuple[int, int]]:
    """Chunk a document into ranges that respect a tokenizer's max token budget.

    Args:
        text: Original text.
        offsets: Token offset mapping (start, end) for each token.
        max_tokens: Maximum tokens allowed per chunk.
        stride: Overlap between chunks to prevent boundary loss.
    """

    if not offsets:
        return [(0, len(text))]

    chunks: list[tuple[int, int]] = []
    start_idx = 0
    while start_idx < len(offsets):
        end_idx = min(len(offsets), start_idx + max_tokens)
        start_char = offsets[start_idx][0]
        end_char = offsets[end_idx - 1][1]
        chunks.append((start_char, end_char))
        if end_idx == len(offsets):
            break
        start_idx = max(end_idx - stride, start_idx + 1)

    return chunks
