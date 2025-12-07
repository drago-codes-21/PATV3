"""Text processing helper functions used across detectors and severity."""

from __future__ import annotations

import unicodedata
from typing import Iterable, Sequence

SENTENCE_DELIMITERS = ".!?\n"

# Characters we are happy to strip from the *edges* of spans.
# NOTE:
#   - We intentionally do NOT include digits or letters here.
#   - Hyphen ("-") is also excluded to avoid trimming IDs like "CUST-1234".
TRIM_CHARS = " \t\r\n\"'`“”‘’[](){}<>….,;:!?|/"


def normalize_text(text: str) -> str:
    """Apply lightweight, reversible normalisation suitable for detector input.

    We use NFKC to pull visually similar unicode into a consistent representation
    so downstream regex/token alignment is stable across pasted email dumps.

    This is intentionally minimal and does NOT:
        - lower-case the text
        - strip whitespace
        - collapse punctuation
    Those are left to the detectors/severity layers where needed.
    """
    if not text:
        return ""
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
    exceeded. This keeps chunks within the embedding/tokenizer budget while
    preserving character offsets.

    Returns:
        A list of (start, end) character index pairs. If `text` is empty,
        returns [(0, 0)].
    """
    if not text:
        return [(0, 0)]

    if max_length <= 0:
        # Defensive: we never want an infinite loop or zero-length segments.
        raise ValueError("max_length must be a positive integer")

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
    """Trim whitespace/punctuation from a span and return adjusted boundaries.

    Args:
        text: The original text.
        start: Proposed start index (may be inside TRIM_CHARS).
        end: Proposed end index (exclusive; may be inside TRIM_CHARS).

    Returns:
        A tuple of (trimmed_start, trimmed_end, trimmed_text).

    Notes:
        - Only leading/trailing characters in TRIM_CHARS are removed.
        - Core content (letters, digits, internal punctuation) is preserved.
        - If the span collapses (start >= end), `trimmed_text` will be "".
    """
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
        text:
            Original text.
        offsets:
            Token offset mapping (start, end) for each token. Must be sorted in
            the order tokens appear in the text.
        max_tokens:
            Maximum tokens allowed per chunk. Must be > 0.
        stride:
            Overlap (in tokens) between adjacent chunks to prevent boundary
            loss. Must be >= 0 and < max_tokens for meaningful behavior.

    Returns:
        A list of (start_char, end_char) character index pairs corresponding
        to contiguous token spans.

    Notes:
        - If `offsets` is empty, we return a single chunk spanning the whole
          text to avoid surprising callers.
        - Character offsets are taken directly from the provided token offsets,
          so alignment with detectors is preserved.
    """
    if not offsets:
        return [(0, len(text))]

    if max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")

    if stride < 0:
        raise ValueError("stride must be non-negative")

    chunks: list[tuple[int, int]] = []
    start_idx = 0
    n_tokens = len(offsets)

    while start_idx < n_tokens:
        end_idx = min(n_tokens, start_idx + max_tokens)
        start_char = offsets[start_idx][0]
        end_char = offsets[end_idx - 1][1]
        chunks.append((start_char, end_char))

        if end_idx == n_tokens:
            break

        # Ensure progress even if stride is misconfigured.
        next_start = end_idx - stride if stride < max_tokens else end_idx - 1
        start_idx = max(next_start, start_idx + 1)

    return chunks
