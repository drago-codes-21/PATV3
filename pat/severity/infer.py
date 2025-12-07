"""Severity-only inference entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from pat.detectors.runner import DetectorRunner
from pat.embeddings import EmbeddingModel, EmbeddingModelError
from pat.fusion import FusionEngine, FusedSpan
from pat.severity.features import (
    FEATURE_NAMES,
    assert_feature_schema,
    compute_neighbor_stats,
    compute_span_sentence_context,
    compute_token_position_ratio,
    extract_span_features,
    get_span_context_text,
    span_features_to_vector,
)
from pat.severity.model import SeverityModel
from pat.utils.text import compute_sentence_boundaries

LOG = logging.getLogger(__name__)


def _ensure_embedding_model(provided: EmbeddingModel | None) -> EmbeddingModel | None:
    if provided is not None:
        return provided
    try:
        return EmbeddingModel()
    except EmbeddingModelError as exc:  # pragma: no cover - fallback
        LOG.warning("Embedding model unavailable: %s", exc)
        return None


def run_severity_inference(
    text: str,
    *,
    runner: DetectorRunner | None = None,
    fusion: FusionEngine | None = None,
    severity_model: SeverityModel | None = None,
    embedding_model: EmbeddingModel | None = None,
    include_features: bool = False,
) -> Dict[str, Any]:
    """Run detectors → fusion → severity feature extraction → severity model."""

    assert_feature_schema(FEATURE_NAMES)
    runner = runner or DetectorRunner()
    fusion = fusion or FusionEngine()
    severity_model = severity_model or SeverityModel()
    embedding_model = _ensure_embedding_model(embedding_model)

    detector_results = runner.run(text)
    fused_spans = fusion.fuse(detector_results, text=text)
    sentences = compute_sentence_boundaries(text)
    neighbor_stats = compute_neighbor_stats(fused_spans)

    spans_out: List[Dict[str, Any]] = []
    for idx, span in enumerate(fused_spans):
        sentence_index, sentence_ratio = compute_span_sentence_context(span, sentences)
        token_ratio = compute_token_position_ratio(text, span)
        span_text = span.text or text[span.start : span.end]

        embedding_vec = np.zeros(1, dtype=float)
        context_vec = np.zeros(1, dtype=float)
        if embedding_model is not None:
            embedding_vec = embedding_model.encode(span_text)
            context_vec = embedding_model.encode(get_span_context_text(text, span.start, span.end))

        neighbor_span_count, neighbor_high_risk_count = neighbor_stats.get(idx, (0, 0))

        features = extract_span_features(
            span,
            text,
            sentence_index=sentence_index,
            sentence_position_ratio=sentence_ratio,
            token_position_ratio=token_ratio,
            embedding=embedding_vec,
            context_embedding=context_vec,
            domain_flags=None,
            neighbor_span_count=neighbor_span_count,
            neighbor_high_risk_count=neighbor_high_risk_count,
            context_metadata=None,
        )
        feature_vector = span_features_to_vector(features)
        score, label, probs = severity_model.predict(feature_vector, pii_type=span.pii_type)
        span.severity_score = score
        span.severity_label = label
        span.severity_probs = probs
        spans_out.append(
            {
                "start": span.start,
                "end": span.end,
                "text": span_text,
                "pii_type": span.pii_type,
                "all_types": list(getattr(span, "all_types", [])),
                "severity_score": score,
                "severity_label": label,
                "detector_sources": [s.detector_name for s in span.sources],
                "probs": probs,
                "features": features if include_features or _debug_enabled() else None,
            }
        )

    return {"text": text, "spans": spans_out}


def _debug_enabled() -> bool:
    return os.getenv("PAT_DEBUG_SEVERITY", "").lower() in {"1", "true", "yes"}


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=str, help="Raw text to analyse.")
    parser.add_argument("--input", type=Path, help="Path to a text file.")
    parser.add_argument("--json-output", type=Path, help="Write JSON output to this path.")
    parser.add_argument("--debug", action="store_true", help="Include feature vectors in output.")
    args = parser.parse_args(argv)

    if not args.text and not args.input:
        raise SystemExit("Provide --text or --input.")
    text = args.text or args.input.read_text(encoding="utf-8")

    logging.basicConfig(level=logging.INFO)
    result = run_severity_inference(text, include_features=args.debug or _debug_enabled())

    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        LOG.info("Wrote JSON output to %s", args.json_output)
    else:
        print("Input text:")
        print(text)
        print("\nDetected spans:")
        for span in result["spans"]:
            print(
                f"- '{span['text']}' [{span['pii_type']}] "
                f"({span['start']}, {span['end']}) "
                f"severity={span['severity_label']} ({span['severity_score']:.3f})"
            )
            if args.debug or _debug_enabled():
                print(f"  features: {span.get('features')}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
