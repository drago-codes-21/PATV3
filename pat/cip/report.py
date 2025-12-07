"""CIP report aggregation and rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _fmt_line(key: str, value: Any) -> str:
    return f"- {key}: {value}"


def render_markdown(payload: Dict[str, Any]) -> str:
    lines: list[str] = ["# PAT Continuous Improvement Pipeline Report", ""]
    lines.append("## Regression Tests")
    lines.append(_fmt_line("status", payload.get("regression_status")))
    lines.append(_fmt_line("details", payload.get("regression_details", "")))

    lines.append("\n## Calibration")
    lines.append(_fmt_line("applied", payload.get("calibration_applied")))
    lines.append(_fmt_line("rejected_types", payload.get("calibration_rejected")))
    lines.append(_fmt_line("thresholds_new", payload.get("thresholds_new")))

    lines.append("\n## Drift & Severity")
    lines.append(_fmt_line("drift_max", payload.get("drift_max")))
    lines.append(_fmt_line("drift_threshold", payload.get("drift_threshold")))
    lines.append(_fmt_line("retrain_status", payload.get("retrain_status")))
    lines.append(_fmt_line("model_path", payload.get("model_path")))

    lines.append("\n## End-to-End Evaluation")
    for tag in ("baseline", "candidate"):
        metrics = payload.get(f"e2e_{tag}") or {}
        lines.append(f"- {tag}: precision={metrics.get('precision')}, recall={metrics.get('recall')}, "
                     f"f1={metrics.get('f1')}, mean_iou={metrics.get('mean_iou')}")

    lines.append("\n## Fusion Health")
    lines.append(_fmt_line("multi_detector_spans", payload.get("multi_detector_spans")))
    lines.append(_fmt_line("single_detector_spans", payload.get("single_detector_spans")))

    lines.append("\n## Config Deltas")
    lines.append(_fmt_line("thresholds_old", payload.get("thresholds_old")))
    lines.append(_fmt_line("thresholds_new", payload.get("thresholds_new")))
    lines.append(_fmt_line("model_old", payload.get("model_old")))
    lines.append(_fmt_line("model_new", payload.get("model_new")))

    lines.append("\n## Suggested Actions")
    for action in payload.get("suggested_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines)


def write_report(payload: Dict[str, Any], md_path: Path, json_path: Path) -> None:
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
