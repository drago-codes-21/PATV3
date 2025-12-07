"""Developer-facing debug harness for PAT pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from pat.pipeline.core import run_pipeline
from pat.policy import PolicyEngine
from pat.severity import SeverityModel

LOG = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=str, help="Raw text to inspect.")
    parser.add_argument("--input", type=Path, help="Input file containing text.")
    parser.add_argument("--policy-config", type=Path, help="Optional policy rules override.")
    parser.add_argument("--severity-model", type=Path, help="Optional severity model override.")
    parser.add_argument("--json-output", type=Path, help="Write full debug output to JSON.")
    parser.add_argument("--trunc-span", type=int, default=64, help="Truncate logged span text to this length.")
    args = parser.parse_args(argv)

    if not args.text and not args.input:
        raise SystemExit("Provide --text or --input.")
    text = args.text or args.input.read_text(encoding="utf-8")

    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("PAT_DEBUG_DETECTORS", "true")
    os.environ.setdefault("PAT_DEBUG_SEVERITY", "true")
    os.environ.setdefault("PAT_DEBUG_POLICY", "true")

    policy = PolicyEngine(policy_rules_path=args.policy_config) if args.policy_config else PolicyEngine()
    severity = SeverityModel(model_path=args.severity_model) if args.severity_model else SeverityModel()

    result = run_pipeline(text, policy_engine=policy, severity_model=severity)

    def _truncate(val: str) -> str:
        return val if len(val) <= args.trunc_span else val[: args.trunc_span] + "..."

    print("=== Input ===")
    print(text)
    print("\n=== Spans ===")
    for span in result.spans:
        print(
            f"[{span.start},{span.end}] {span.pii_type} sev={getattr(span,'severity_label',None)} "
            f"rule={getattr(span, 'policy_decision', {}).get('rule_id') if hasattr(span,'policy_decision') else None} "
            f"text='{_truncate(span.text)}'"
        )
    print("\n=== Sanitized ===")
    print(result.sanitized_text)

    if args.json_output:
        payload = {
            "text": text,
            "sanitized_text": result.sanitized_text,
            "decision": result.decision,
            "severity_label": result.severity_label,
            "severity_score": result.severity_score,
            "spans": [
                {
                    "start": s.start,
                    "end": s.end,
                    "pii_type": s.pii_type,
                    "severity_label": getattr(s, "severity_label", None),
                    "severity_probs": getattr(s, "severity_probs", None),
                    "policy_decision": getattr(s, "policy_decision", None),
                    "text": s.text if args.trunc_span <= 0 else _truncate(s.text),
                }
                for s in result.spans
            ],
        }
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOG.info("Wrote debug JSON to %s", args.json_output)


if __name__ == "__main__":  # pragma: no cover
    main()
