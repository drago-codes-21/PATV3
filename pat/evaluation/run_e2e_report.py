"""Run end-to-end evaluation and emit markdown/JSON reports."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from pat.pipeline.pipeline import RedactionPipeline

LOG = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def _evaluate_sample(pred_spans, gold_spans):
    tp = fp = fn = 0
    boundary_ious: list[float] = []
    for pred in pred_spans:
        best_iou = 0.0
        best_gold = None
        for gold in gold_spans:
            if gold.get("pii_type") != pred.pii_type:
                continue
            iou = _iou(pred.start, pred.end, gold["start"], gold["end"])
            if iou > best_iou:
                best_iou, best_gold = iou, gold
        if best_iou >= 0.5:
            tp += 1
            boundary_ious.append(best_iou)
            gold_spans = [g for g in gold_spans if g is not best_gold]
        else:
            fp += 1
    fn = len(gold_spans)
    return tp, fp, fn, boundary_ious


def _severity_distribution(spans) -> dict[str, int]:
    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "VERY_HIGH": 0}
    for span in spans:
        label = getattr(span, "severity_label", None)
        if label in counts:
            counts[label] += 1
    return counts


def _render_markdown(metrics: dict[str, Any]) -> str:
    lines = ["# End-to-End Evaluation Report", ""]
    lines.append(f"- Samples: {metrics['samples']}")
    lines.append(f"- Precision: {metrics['precision']:.3f}")
    lines.append(f"- Recall: {metrics['recall']:.3f}")
    lines.append(f"- F1: {metrics['f1']:.3f}")
    lines.append(f"- Boundary IoU (mean): {metrics['mean_iou']:.3f}")
    lines.append("\n## Severity Distribution")
    for label, count in metrics["severity_distribution"].items():
        lines.append(f"- {label}: {count}")
    lines.append("\n## Error Buckets")
    for bucket, count in metrics["error_buckets"].items():
        lines.append(f"- {bucket}: {count}")
    lines.append("\n## Detector Agreement")
    lines.append(f"- Multi-detector spans: {metrics['multi_detector_spans']}")
    lines.append(f"- Single-detector spans: {metrics['single_detector_spans']}")
    return "\n".join(lines)


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL with text + spans annotations.")
    parser.add_argument("--output-json", type=Path, default=Path("e2e_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("e2e_report.md"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    rows = _load_jsonl(args.input)
    pipeline = RedactionPipeline()

    tp = fp = fn = 0
    all_ious: list[float] = []
    severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "VERY_HIGH": 0}
    error_buckets = {"missed": 0, "over_detect": 0, "boundary_miss": 0}
    multi_detector_spans = single_detector_spans = 0

    for row in rows:
        text = row.get("text") or ""
        gold = row.get("spans") or []
        output = pipeline.run(text, context=row.get("context"))
        fused_spans = output["pii_spans"]

        tpi, fpi, fni, ious = _evaluate_sample(list(fused_spans), list(gold))
        tp += tpi
        fp += fpi
        fn += fni
        all_ious.extend(ious)

        for span in fused_spans:
            if len(span.sources) > 1:
                multi_detector_spans += 1
            else:
                single_detector_spans += 1
            label = span.severity_label
            if label in severity_counts:
                severity_counts[label] += 1

        if fni:
            error_buckets["missed"] += fni
        if fpi:
            error_buckets["over_detect"] += fpi
        if ious and np.mean(ious) < 0.7:
            error_buckets["boundary_miss"] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    metrics: Dict[str, Any] = {
        "samples": len(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "severity_distribution": severity_counts,
        "error_buckets": error_buckets,
        "multi_detector_spans": multi_detector_spans,
        "single_detector_spans": single_detector_spans,
    }

    args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    args.output_md.write_text(_render_markdown(metrics), encoding="utf-8")
    LOG.info("Wrote reports to %s and %s", args.output_json, args.output_md)


if __name__ == "__main__":  # pragma: no cover
    run()
