"""Structured debug logging for detector/fusion/severity decisions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from pat.config import get_settings

LOG = logging.getLogger("pat.detectors.debug")


def is_enabled() -> bool:
    try:
        return bool(get_settings().debug_detectors)
    except Exception:
        return False


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
    """Emit a JSONL debug line when detector debugging is enabled."""

    if not is_enabled():
        return

    payload = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "detector_name": detector_name,
        "action": action,
        "text_window": text_window,
        "span_before": span_before,
        "span_after": span_after,
        "score": score,
        "details": details or {},
    }
    LOG.info(json.dumps(payload, ensure_ascii=False))
