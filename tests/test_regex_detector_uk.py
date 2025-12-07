import pytest

from pat.detectors.regex_detector import RegexDetector


@pytest.fixture
def detector():
    return RegexDetector()


def test_valid_sort_code_compact(detector):
    text = "Sort code: 200000"
    results = detector.run(text)
    assert any(r.pii_type == "SORT_CODE" for r in results)


def test_valid_account_number_with_keyword(detector):
    text = "Account no: 12345678 for payment"
    results = detector.run(text)
    assert any(r.pii_type == "BANK_ACCOUNT" for r in results)


def test_invalid_account_number_without_keyword(detector):
    text = "Random digits 12345678 should not be treated as account"
    assert detector.run(text) == []


def test_valid_card_number_with_context(detector):
    text = "Card number 4111 1111 1111 1111 is on file."
    results = detector.run(text)
    assert any(r.pii_type == "CARD_NUMBER" for r in results)

def test_card_number_without_keyword_but_valid_luhn(detector):
    text = "my card number is 4539 4512 0398 7356"
    results = detector.run(text)
    assert any(r.pii_type == "CARD_NUMBER" for r in results)


def test_card_number_keyword_relaxed(detector):
    text = "my card number is 5555-5555-5555-4444"
    results = detector.run(text)
    assert any(r.pii_type == "CARD_NUMBER" for r in results)

def test_card_like_product_id_not_detected(detector):
    text = "Product id 4111-1111-1111-1111 should be ignored."
    assert detector.run(text) == []


def test_valid_gb_iban(detector):
    text = "IBAN GB29 NWBK 6016 1331 9268 19 is valid."
    results = detector.run(text)
    assert any(r.pii_type in {"BANK_ACCOUNT", "IBAN"} for r in results)


def test_invalid_gb_iban_checksum(detector):
    text = "IBAN GB29 NWBK 6016 1331 9268 18 is invalid."
    assert detector.run(text) == []


def test_building_society_roll_number(detector):
    text = "Roll number: ABC-1234/9XZ"
    results = detector.run(text)
    assert any(r.pii_type == "CUSTOMER_ID" for r in results)


def test_roll_number_requires_keyword(detector):
    text = "ABC-1234/9XZ should not match without roll context."
    assert detector.run(text) == []


def test_generic_text_no_false_positive(detector):
    text = "Invoice 2023-12-34 and reference 99887766 should not trigger banking regex."
    assert detector.run(text) == []


def test_phone_detected_with_keyword_and_two_numbers(detector):
    text = "I want to change my number from (+91)7774927989 to (+91)9766264652"
    results = [r for r in detector.run(text) if r.pii_type == "PHONE"]
    assert len(results) == 2


def test_phone_not_detected_without_keyword(detector):
    text = "Reach me at +44 7700 900123 later."
    results = [r for r in detector.run(text) if r.pii_type == "PHONE"]
    assert not results


def test_nino_detection(detector):
    text = "My NI number is AB123456C."
    results = [r for r in detector.run(text) if r.pii_type == "NI_NUMBER"]
    assert results


def test_nhs_number_checksum(detector):
    text_valid = "NHS number 943 476 5919 is valid."
    text_invalid = "NHS number 943 476 5918 is invalid."
    assert any(r.pii_type == "NHS_NUMBER" for r in detector.run(text_valid))
    assert not any(r.pii_type == "NHS_NUMBER" for r in detector.run(text_invalid))


def test_postcode_detection(detector):
    text = "The address is 221B Baker Street, London NW1 6XE."
    assert any(r.pii_type == "POSTCODE" for r in detector.run(text))


def test_ipv4_detection(detector):
    text = "Connect to 192.168.10.25 for diagnostics."
    assert any(r.pii_type == "IP_ADDRESS" for r in detector.run(text))


def test_ipv6_detection(detector):
    text = "Reach the service at fe80::1ff:fe23:4567:890a over IPv6."
    assert any(r.pii_type == "IPV6_ADDRESS" for r in detector.run(text))


def test_student_id_detection(detector):
    text = "Student ID SID374527 is assigned."
    assert any(r.pii_type == "STUDENT_ID" for r in detector.run(text))


def test_device_id_detection(detector):
    text = "Using device dev-4865530102f5 for this test."
    assert any(r.pii_type == "DEVICE_ID" for r in detector.run(text))


def test_api_key_detection(detector):
    text = "Here is the API key: API-5d4e01549263af05bb1a1044."
    assert any(r.pii_type == "API_KEY" for r in detector.run(text))


def test_money_detection(detector):
    text = "Paid £1591 yesterday."
    assert any(r.pii_type == "MONEY" for r in detector.run(text))
