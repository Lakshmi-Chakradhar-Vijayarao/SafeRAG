"""
SafeRAG System Test Suite

Automated tests validating:
- Claim-level verification
- Hallucination detection
- Safety gating
- Fail-safe behavior
- Determinism
- Audit logging

Each test maps directly to TEST_BENCH.md.
"""

import sys
from pathlib import Path

# --------------------------------------------------
# Ensure project root is on PYTHONPATH
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pytest
from saferag_bootstrap import bootstrap
from app.service import run_saferag
from app.schemas import SafeRAGRequest


# --------------------------------------------------
# Global test setup
# --------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def setup_saferag():
    """
    Initialize SafeRAG once for all tests.
    """
    bootstrap()


# --------------------------------------------------
# Test Category 1 — Supported Claims
# --------------------------------------------------

def test_supported_claim_accept():
    """Fully supported factual claim should be ACCEPTED."""
    req = SafeRAGRequest(
        request_id="test_supported",
        generated_text="Metformin is the first line treatment for type 2 diabetes."
    )

    decision, claims, metrics = run_saferag(req)

    assert decision == "ACCEPT"
    assert len(claims) == 1
    assert claims[0]["label"] == "SUPPORTED"
    assert metrics["support_rate"] == 1.0


# --------------------------------------------------
# Test Category 2 — Unsupported Claims
# --------------------------------------------------

def test_unsupported_claim_refuse():
    """Unsupported medical advice must be REFUSED or REJECTED."""
    req = SafeRAGRequest(
        request_id="test_unsupported",
        generated_text="ACE inhibitors and ARBs should always be combined."
    )

    decision, claims, _ = run_saferag(req)

    assert decision in {"REFUSE", "REJECT"}
    assert any(c["label"] != "SUPPORTED" for c in claims)


# --------------------------------------------------
# Test Category 3 — Explicit Contradictions
# --------------------------------------------------

def test_contradiction_reject():
    """Semantic contradictions must trigger REJECT."""
    req = SafeRAGRequest(
        request_id="test_contradiction",
        generated_text="Insulin is never used for type 2 diabetes."
    )

    decision, claims, _ = run_saferag(req)

    assert decision == "REJECT"
    assert any(c["label"] == "CONTRADICTED" for c in claims)


# --------------------------------------------------
# Test Category 4 — Mixed Claims
# --------------------------------------------------

def test_mixed_claims_refuse():
    """Mixed supported + contradicted claims must not partially pass."""
    req = SafeRAGRequest(
        request_id="test_mixed",
        generated_text=(
            "Metformin is first line treatment "
            "and insulin is never used for type 2 diabetes."
        )
    )

    decision, claims, _ = run_saferag(req)

    assert decision in {"REFUSE", "REJECT"}
    assert len(claims) >= 2
    assert any(c["label"] == "SUPPORTED" for c in claims)
    assert any(c["label"] != "SUPPORTED" for c in claims)


# --------------------------------------------------
# Test Category 5 — Long Inputs
# --------------------------------------------------

def test_long_input_bounded():
    """Excessively long input must not crash or hang."""
    long_text = "Metformin is first line treatment. " * 100

    req = SafeRAGRequest(
        request_id="test_long",
        generated_text=long_text
    )

    decision, claims, _ = run_saferag(req)

    assert decision in {"REFUSE", "ACCEPT"}
    assert isinstance(claims, list)


# --------------------------------------------------
# Test Category 6 — Nonsense Inputs
# --------------------------------------------------

def test_gibberish_refuse():
    """Low-signal gibberish input must be REFUSED."""
    req = SafeRAGRequest(
        request_id="test_gibberish",
        generated_text="asdkjh qweoiu zxcmn qweqwe"
    )

    decision, claims, _ = run_saferag(req)

    assert decision == "REFUSE"
    assert claims == []


# --------------------------------------------------
# Test Category 7 — Empty Input
# --------------------------------------------------

def test_empty_input_refuse():
    """Empty input must be handled safely."""
    req = SafeRAGRequest(
        request_id="test_empty",
        generated_text=""
    )

    decision, claims, _ = run_saferag(req)

    assert decision == "REFUSE"
    assert claims == []


# --------------------------------------------------
# Test Category 8 — Determinism
# --------------------------------------------------

def test_deterministic_execution():
    """Identical inputs must produce identical outputs."""
    req = SafeRAGRequest(
        request_id="test_determinism",
        generated_text="Metformin is first line treatment."
    )

    out1 = run_saferag(req)
    out2 = run_saferag(req)

    assert out1 == out2


# --------------------------------------------------
# Test Category 9 — Audit Logging
# --------------------------------------------------

def test_audit_log_written():
    """Every request must generate an audit entry."""
    req = SafeRAGRequest(
        request_id="test_audit",
        generated_text="Metformin is first line treatment."
    )

    run_saferag(req)

    log_path = Path("logs/saferag_audit.jsonl")
    assert log_path.exists()

    with open(log_path) as f:
        logs = f.read()

    assert "test_audit" in logs
