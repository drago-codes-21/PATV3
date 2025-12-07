"""
Manual validation runner when pytest is unavailable.

Usage:
    python tests/run_manual_validations.py
"""

from pat.severity.features import FEATURE_NAMES, assert_feature_schema, span_features_to_vector
from tests.detector_layer_validation import run_examples


def main():
    # Schema check
    assert_feature_schema(FEATURE_NAMES)
    vec = span_features_to_vector({name: 0.0 for name in FEATURE_NAMES})
    assert vec.shape == (len(FEATURE_NAMES),)

    # Detector regression harness
    results = run_examples()
    assert results, "Validation harness returned no results."


if __name__ == "__main__":  # pragma: no cover
    main()
