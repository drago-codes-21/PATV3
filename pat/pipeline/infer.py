"""Convenient, single-call inference entrypoint for PAT."""

from __future__ import annotations

import logging
from typing import Any, Dict

from pat.pipeline.core import run_pipeline

LOG = logging.getLogger(__name__)


def run_inference(text: str, *, context: dict[str, Any] | None = None, debug: bool = False) -> Dict[str, Any]:
    """
    Run the full PII redaction pipeline on a single text input.

    Returns a serialisable dict containing sanitized_text, spans, and applied rules.
    """
    if debug:
        logging.basicConfig(level=logging.INFO)
    try:
        result = run_pipeline(text, context=context or {})
    except Exception as exc:  # pragma: no cover - defensive
        LOG.exception("PAT inference failed")
        return {
            "sanitized_text": text,
            "error": str(exc),
            "spans": [],
            "decision": "ERROR",
        }

    spans = [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "pii_type": s.pii_type,
            "severity_label": getattr(s, "severity_label", None),
            "severity_score": getattr(s, "severity_score", None),
            "policy_decision": getattr(s, "policy_decision", None),
        }
        for s in result.spans
    ]

    return {
        "sanitized_text": result.sanitized_text,
        "spans": spans,
        "decision": result.decision,
        "severity_label": result.severity_label,
        "severity_score": result.severity_score,
        "applied_rules": result.applied_rules,
    }


__all__ = ["run_inference"]
