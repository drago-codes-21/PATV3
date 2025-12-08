from pathlib import Path

import numpy as np

from pat.pipeline import RedactionPipeline
from pat.fusion import FusionEngine
from pat.policy import PolicyEngine
from pat.detectors.runner import DetectorRunner


class _ZeroEmbeddings:
    def encode_batch(self, texts):
        return np.zeros((len(texts), 4), dtype=float)

    def encode(self, text):
        return np.zeros(4, dtype=float)


class _StaticSeverity:
    def __init__(self, score: float = 0.82, label: str = "HIGH") -> None:
        self.model = True
        self.score = score
        self.label = label

    def predict(self, feature_vector, pii_type=None):
        return self.score, self.label, {self.label: 0.9}


RAW_TEXT = """\
Subject: Internal Audit – Mixed PII & Secrets Review (Raw)

Customer contact:
Name: Emily Johnson
Phone: +44 7700 900111
Emergency Contact: Rohan Patel, +44 7700 811222
Email: emily.johnson@example.com

Address:
2B Rosewood Court,
Apartment 12,
Dublin D02 X285,
Ireland.
Shipping: shipping_postcode="D02 X285"

CRM:
Customer Display ID: CUST-00043219
Loyalty ID: LOY-9988-7766
Legacy Numeric ID: 10492837
Wohnung 7,
10115 Berlin,
Germany.
Tax ID (DE): 12 345 678 901
Passport: XJ552993847

Security:
INTERNAL_JWT_SECRET="#SuperSecret#2024!"
SMTP_PASSWORD='Email!Pass2024'

Health:
Health notes: chronic migraine and generalized anxiety.

Logs:
dob="1988-07-22"
"dob": "1992-12-31"
DOB 14/03/1985
payload={"full_name": "Emily Johnson", "national_id": "QQ 12 34 56 C"}

Repo mention: committed to a private git repo.
Counters:
total_emails_sent: 3456
unique_users: 789
"""


def _run_pipeline(text: str) -> str:
    pipeline = RedactionPipeline(
        detector_runner=DetectorRunner(),
        fusion_engine=FusionEngine(),
        policy_engine=PolicyEngine(),
        severity_model=_StaticSeverity(),
        embedding_model=_ZeroEmbeddings(),
    )
    return pipeline.run(text)["sanitized_text"]


def test_refined_detection_and_masking_behaviour():
    sanitized = _run_pipeline(RAW_TEXT)

    # Phones fully masked (no raw numbers or stray '+')
    assert "+44 7700 900111" not in sanitized
    assert "+44 7700 811222" not in sanitized
    assert "+<CONTACT>" not in sanitized

    # DoBs masked
    for dob in ("1988-07-22", "1992-12-31", "14/03/1985"):
        assert dob not in sanitized

    # Addresses and postcodes masked
    assert "Dublin D02 X285" not in sanitized
    assert "shipping_postcode=\"D02 X285\"" not in sanitized
    assert "10115 Berlin" not in sanitized
    assert "Wohnung 7" not in sanitized

    # Government/tax IDs masked fully
    assert "12 345 678 901" not in sanitized
    assert "XJ552993847" not in sanitized
    assert "QQ 12 34 56 C" not in sanitized

    # Secrets fully masked without suffix leakage
    assert "#SuperSecret#2024!" not in sanitized
    assert "Email!Pass2024" not in sanitized

    # Health conditions masked
    assert "migraine" not in sanitized.lower()
    assert "anxiety" not in sanitized.lower()

    # Counters should remain visible
    assert "total_emails_sent: 3456" in sanitized
    assert "unique_users: 789" in sanitized

    # Git repo should not be tagged as a person
    assert "git repo" in sanitized
    assert "<PERSON> repo" not in sanitized

    # Full names masked as a whole
    assert "Emily Johnson" not in sanitized
    assert "\"full_name\": \"<PERSON>\"" in sanitized
