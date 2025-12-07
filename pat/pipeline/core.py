"""Core pipeline orchestration helpers for PAT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pat.detectors import DetectorRunner
from pat.embeddings import EmbeddingModel
from pat.fusion import FusionEngine, FusedSpan
from pat.pipeline.pipeline import RedactionPipeline
from pat.policy import PolicyEngine
from pat.severity import SeverityModel


@dataclass
class PipelineResult:
    """Structured output of a PAT pipeline run."""

    original_text: str
    sanitized_text: str
    decision: str
    severity_label: str
    severity_score: float
    spans: list[FusedSpan]
    raw_detector_results: list
    applied_rules: list[str]


def run_pipeline(
    text: str,
    *,
    context: dict[str, Any] | None = None,
    detector_runner: DetectorRunner | None = None,
    fusion_engine: FusionEngine | None = None,
    severity_model: SeverityModel | None = None,
    policy_engine: PolicyEngine | None = None,
    embedding_model: EmbeddingModel | None = None,
) -> PipelineResult:
    """Execute PAT end-to-end and return a structured result."""

    pipeline = RedactionPipeline(
        detector_runner=detector_runner,
        fusion_engine=fusion_engine,
        severity_model=severity_model,
        policy_engine=policy_engine,
        embedding_model=embedding_model,
    )
    result = pipeline.run(text, context=context)
    return PipelineResult(
        original_text=text,
        sanitized_text=result["sanitized_text"],
        decision=result["decision"],
        severity_label=result["severity_label"],
        severity_score=result["severity_score"],
        spans=result["pii_spans"],
        raw_detector_results=result["raw_detector_results"],
        applied_rules=result["applied_rules"],
    )
