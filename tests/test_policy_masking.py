from pat.policy.masking import apply_mask


def test_partial_email_masks_local():
    masked = apply_mask("partial_email", "john.doe@example.com", {"mask_char": "*", "keep_domain": True})
    assert "example.com" in masked
    assert "john" not in masked


def test_preserve_last_n():
    masked = apply_mask("preserve_last_n", "123456789", {"last_n": 4, "mask_char": "X"})
    assert masked.endswith("6789")
    assert masked.startswith("XXXXX")


def test_preserve_format_keeps_separators():
    masked = apply_mask("preserve_format", "4111-1111-1111-1111", {"mask_char": "X"})
    assert masked.count("-") == 3
    assert set(masked.replace("-", "")) == {"X"}


def test_placeholder_replacement():
    masked = apply_mask("placeholder", "secret", {"placeholder": "[REDACTED]"})
    assert masked == "[REDACTED]"
