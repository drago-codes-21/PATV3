"""Structured debug logging for detector/fusion/severity decisions."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Mapping

from pat.config import get_settings

LOG = logging.getLogger("pat.detectors.debug")

# Hard cap on how much context we emit per event to keep logs manageable.
_MAX_TEXT_WINDOW = 200


def _settings_debug_enabled() -> bool:
    """Check settings flags for any debug switches that should emit logs."""
    try:
        settings = get_settings()
    except Exception:
        return False

    # Support multiple debug knobs; any of them can turn on structured logging.
    flags = [
        getattr(settings, "debug_detectors", False),
        getattr(settings, "debug_fusion", False),
    ]
    return any(bool(f) for f in flags)


def _env_debug_enabled() -> bool:
    """Allow environment variable overrides for ad-hoc debugging."""
    # Examples:
    #   PAT_DEBUG_DETECTORS=1
    #   PAT_DEBUG_POLICY=true
    val = os.getenv("PAT_DEBUG_DETECTORS", "") or os.getenv("PAT_DEBUG_ALL", "")
    return val.lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """Return True if structured debug logging should be emitted."""
    return _settings_debug_enabled() or _env_debug_enabled()


def _json_safe(obj: Any) -> Any:
    """Make arbitrary objects JSON-serialisable in a lossy but safe way."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    # Fallback to string representation for unknown types (e.g. spaCy spans, numpy arrays)
    return repr(obj)


def log_decision(
    *,
    detector_name: str,
    action: str,
    text_window: str,
    span_before: tuple[int, int] | None = None,
    span_after: tuple[int, int] | None = None,
    score: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Emit a JSONL debug line when debugging is enabled.

    All logs are:
        - timestamped in UTC
        - JSON-serialisable
        - bounded in size (text_window truncated)
    """
    if not is_enabled():
        return

    window = text_window or ""
    if len(window) > _MAX_TEXT_WINDOW:
        window = window[:_MAX_TEXT_WINDOW] + "…"

    payload = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "detector_name": detector_name,
        "action": action,
        "text_window": window,
        "span_before": span_before,
        "span_after": span_after,
        "score": float(score) if score is not None else None,
        "details": _json_safe(details or {}),
    }

    try:
        LOG.info(json.dumps(payload, ensure_ascii=False))
    except Exception as exc:
        # As a last resort, fall back to repr; logging must never crash the pipeline.
        LOG.error("Failed to serialise debug payload for %s: %s", detector_name, exc)
        LOG.info("DEBUG_FALLBACK %r", payload)
