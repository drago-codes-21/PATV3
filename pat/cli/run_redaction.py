"""CLI for running PAT redaction pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from pat.pipeline import RedactionPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Text to redact. If omitted, read from stdin.")
    parser.add_argument("--channel", help="Channel metadata (e.g. EMAIL_OUTBOUND).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text = args.text or sys.stdin.read()
    if not text:
        print("No text provided.", file=sys.stderr)
        return 1

    context: Dict[str, Any] = {}
    if args.channel:
        context["channel"] = args.channel

    pipeline = RedactionPipeline()
    result = pipeline.run(text, context)
    serialisable = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "pii_spans",
            "raw_detector_results",
        }
    }
    serialisable["pii_spans"] = [
        {
            "start": span.start,
            "end": span.end,
            "text": span.text,
            "pii_type": span.pii_type,
            "confidence": span.confidence,
            "severity_score": span.severity_score,
            "severity_label": span.severity_label,
        }
        for span in result["pii_spans"]
    ]

    print(json.dumps(serialisable, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
