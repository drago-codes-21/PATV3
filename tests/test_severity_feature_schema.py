from pat.severity.features import FEATURE_NAMES, assert_feature_schema, span_features_to_vector


def test_feature_schema_length_and_order():
    assert len(FEATURE_NAMES) == 130
    assert isinstance(FEATURE_NAMES, list)
    # Ensure order is stable by round-tripping through the assertion helper.
    assert_feature_schema(FEATURE_NAMES)


def test_span_features_to_vector_enforces_schema():
    # Build a minimal feature dict with required keys
    feature_dict = {name: 0.0 for name in FEATURE_NAMES}
    vec = span_features_to_vector(feature_dict)
    assert vec.shape == (len(FEATURE_NAMES),)

    # Removing a key should raise
    bad = feature_dict.copy()
    bad.pop(FEATURE_NAMES[0])
    import pytest

    with pytest.raises(AssertionError):
        span_features_to_vector(bad)
