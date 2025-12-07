"""CLI entrypoint for running the PAT pipeline on a text file."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import os

# These imports will now succeed as the modules have been implemented.
from pat.embeddings import EmbeddingModel
from pat.pipeline.core import run_pipeline 
from pat.policy.engine import PolicyEngine
from pat.severity.model import SeverityModel

LOG = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to input text file.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write sanitized text.")
    parser.add_argument("--policy-config", type=Path, help="Override policy rules file.")
    parser.add_argument("--severity-model", type=Path, help="Override severity model path.")
    parser.add_argument("--debug", action="store_true", help="Enable detector/policy debug logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    text = args.input.read_text(encoding="utf-8")
    if args.debug:
        os.environ["PAT_DEBUG_POLICY"] = "true"
        os.environ.setdefault("PAT_DEBUG_DETECTORS", "true")
        os.environ.setdefault("PAT_DEBUG_SEVERITY", "true")

    policy = PolicyEngine(policy_rules_path=args.policy_config) if args.policy_config else PolicyEngine()
    severity = SeverityModel(model_path=args.severity_model) if args.severity_model else SeverityModel()
    embeddings = EmbeddingModel()

    result = run_pipeline(text, policy_engine=policy, severity_model=severity, embedding_model=embeddings)
    args.output.write_text(result.sanitized_text, encoding="utf-8")
    LOG.info("Sanitized text written to %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
