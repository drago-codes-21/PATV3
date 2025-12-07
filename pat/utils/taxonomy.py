"""Utilities for loading and working with the canonical PII taxonomy."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable

from pat.config import get_settings


@lru_cache(maxsize=1)
def _load_schema() -> dict[str, Any]:
    settings = get_settings()
    schema_path = settings.project_root / "pat" / "config" / "pii_schema.json"
    if not schema_path.exists():
        return {"categories": [], "type_priority": {}, "placeholders": {}}
    return json.loads(schema_path.read_text(encoding="utf-8"))


def pii_schema() -> dict[str, Any]:
    """Return the loaded taxonomy schema."""

    return _load_schema()


@lru_cache(maxsize=None)
def type_to_category_map() -> Dict[str, str]:
    """Map pii_type -> category from the schema."""

    mapping: Dict[str, str] = {}
    for category in _load_schema().get("categories", []):
        cat_id = str(category.get("id", "")).upper()
        for pii_type in category.get("pii_types", []):
            mapping[str(pii_type).upper()] = cat_id
    return mapping


def category_for_type(pii_type: str | None) -> str | None:
    """Resolve the high-level category for a pii_type."""

    if not pii_type:
        return None
    return type_to_category_map().get(str(pii_type).upper())


def placeholder_for_type(pii_type: str) -> str | None:
    """Return semantic placeholder for a pii_type if present."""

    if not pii_type:
        return None
    placeholders = _load_schema().get("placeholders", {})
    return placeholders.get(str(pii_type).upper())


def priority_for_type(pii_type: str, default: int = 0) -> int:
    """Return ordering priority for conflict resolution."""

    priorities = _load_schema().get("type_priority", {})
    return int(priorities.get(str(pii_type).upper(), default))


def categories_for_types(pii_types: Iterable[str]) -> set[str]:
    """Return set of categories for a collection of pii types."""

    return {category_for_type(t) for t in pii_types if category_for_type(t)}


__all__ = [
    "pii_schema",
    "category_for_type",
    "categories_for_types",
    "placeholder_for_type",
    "priority_for_type",
    "type_to_category_map",
]
