"""
Manual validation harness for the detector layer (detectors -> fusion -> severity).

Run with: `python tests/detector_layer_validation.py`
"""

from __future__ import annotations

import pprint

from pat.pipeline import RedactionPipeline

SAMPLE_TEXTS = {
    "card": "My card number is 4111 1111 1111 1111 exp 12/25.",
    "bank": "Please use sort code 20-00-00 and account number 12345678 for the transfer.",
    "contact": "Email me at analyst@example.co.uk or call +44 7700 900123 tomorrow.",
    "credential": "The one time password is 123456 for your account.",
    "address": "Ship to 10 Downing Street, London SW1A 2AA as discussed.",
    "adjacent_email_bank": "Account 12345678,user@example.com should not merge.",
    "punctuated_phone": "Reach me on (+44) 7700-900-123!!! ASAP.",
    "messy_credential": "pwd: 987654 was sent via chat, please reset asap.",
    "long_iban": "IBAN GB29 NWBK 6016 1331 9268 19 is provided; email foo@bar.com nearby.",
    "card_suffix": "Refund to card ending in 4321 please.",
    "customer_id": "Customer ID is CUST-88765 in our system.",
    "shipping_address": "We have your shipping address on file as 123 Baker Street, London, W1U 6AA.",
}


def run_examples() -> dict[str, dict]:
    pipeline = RedactionPipeline()
    results: dict[str, dict] = {}
    for name, text in SAMPLE_TEXTS.items():
        output = pipeline.run(text, context={"channel": "EMAIL_OUTBOUND"})
        spans = [
            {"pii_type": span.pii_type, "start": span.start, "end": span.end, "text": span.text}
            for span in output["pii_spans"]
        ]
        # Trim assertions
        for span in spans:
            assert span["text"] == span["text"].strip()
        # Fusion separation for adjacent email+bank
        if name == "adjacent_email_bank":
            assert len(spans) >= 2
            types = {s["pii_type"] for s in spans}
            assert "BANK_ACCOUNT" in types and "EMAIL" in types
        # Punctuated phone detection
        if name == "punctuated_phone":
            assert any("PHONE" in s["pii_type"] for s in spans)
        # Long IBAN preserved and email not merged
        if name == "long_iban":
            assert any("BANK_ACCOUNT" in s["pii_type"] for s in spans)
            assert any("EMAIL" in s["pii_type"] for s in spans)
        if name == "card_suffix":
            assert any("CARD_NUMBER" in s["pii_type"] for s in spans)
            assert not any("DATE" in s["pii_type"] for s in spans)
        if name == "customer_id":
            assert any("CUSTOMER_ID" in s["pii_type"] for s in spans)
        if name == "shipping_address":
            # address and postcode should be captured; no credential contamination
            assert any("ADDRESS" in s["pii_type"] for s in spans)
            assert any("POSTCODE" in s["pii_type"] for s in spans)
            assert not any("CREDENTIAL" in s["pii_type"] for s in spans)

        results[name] = {
            "input": text,
            "pii_spans": spans,
            "decision": output["decision"],
            "severity": output["severity_label"],
        }
    return results


if __name__ == "__main__":  # pragma: no cover - manual harness
    result_map = run_examples()
    pprint.pp(result_map)
