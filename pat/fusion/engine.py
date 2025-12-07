"""
Deprecated shim that delegates to the canonical FusionEngine.

This module exists only for backward compatibility with legacy imports such as:
    from pat.fusion.engine import FusionEngine

The maintained implementation lives in:
    pat.fusion.fusion_engine   → FusionEngine
    pat.fusion.span            → FusedSpan
"""

from __future__ import annotations

import warnings

from pat.fusion.fusion_engine import FusionEngine
from pat.fusion.span import FusedSpan

# Emit a deprecation warning once per interpreter session.
warnings.warn(
    (
        "pat.fusion.engine is deprecated and will be removed in a future version. "
        "Import FusionEngine from pat.fusion.fusion_engine and FusedSpan from "
        "pat.fusion.span instead."
    ),
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["FusionEngine", "FusedSpan"]
