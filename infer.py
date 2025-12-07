"""
Inference script to run the PII redaction pipeline on a given text and
output severity scores for each detected span.
"""

import argparse
import logging

from pat.pipeline import RedactionPipeline

# Configure logging to show informational messages from the pipeline.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOG = logging.getLogger(__name__)


def run_inference(text: str):
    """
    Runs the PII redaction pipeline on a given text and prints the
    severity scores for each span.

    This function instantiates the full pipeline, which automatically loads
    the offline embedding model and the trained LightGBM severity model
    from the paths specified in the configuration.

    Args:
        text: The input text to analyze.
    """
    LOG.info("Initializing the redaction pipeline...")
    pipeline = RedactionPipeline()
    LOG.info("Pipeline initialized. Running inference on the provided text...")

    results = pipeline.run(text)

    LOG.info("Inference complete. Found %d PII spans.", len(results.get("pii_spans", [])))

    span_details = results.get("pii_spans", [])

    if not span_details:
        print("\nNo PII spans were detected in the text.")
        return

    print("\n--- Severity Scores for Detected Spans ---")
    for span in span_details:
        print(
            f'  - Span: "{span.text}"\n'
            f"    Type: {span.pii_type}\n"
            f"    Location: ({span.start}, {span.end})\n"
            f"    Severity Score: {span.severity_score:.4f}\n"
            f"    Severity Label: {span.severity_label}\n"
        )


def main():
    """Main entrypoint for the inference script."""
    parser = argparse.ArgumentParser(description="Run PII severity inference on a given text.")
    parser.add_argument("input_text", type=str, help="The input text to be analyzed for PII.")
    args = parser.parse_args()

    run_inference(args.input_text)


if __name__ == "__main__":
    main()