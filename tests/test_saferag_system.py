"""
SafeRAG System Test Suite

Validates:
- Claim truth classification
- Safety gating
- Hallucination containment
- Fail-safe behavior
- Determinism
- Audit logging
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pytest
from saferag_bootstrap import bootstrap
from app.service import run_saferag
from app.schemas import SafeRAGRequest


@pytest.fixture(scope="session", autouse=True)
def setup_saferag():
    bootstrap()


# --------------------------------------------------
# Supported / Verified claims
# --------------------------------------------------

def test_supported_claim_accept():
    req = SafeRAGRequest(
        request_id="test_supported",
        generated_text="Metformin is the first line treatment for type 2 diabetes."
    )

    decision, claims, metrics = run_saferag(req)

    assert decision == "ACCEPT"
    assert len(claims) == 1
    assert claims[0]["label"] == "VERIFIED"
    assert metrics["support_rate"] == 1.0


# --------------------------------------------------
# Unsupported claims
# --------------------------------------------------

def test_unsupported_claim_refuse():
    req = SafeRAGRequest(
        request_id="test_unsupported",
        generated_text="ACE inhibitors and ARBs should always be combined."
    )

    decision, claims, _ = run_saferag(req)

    assert decision in {"REFUSE", "REJECT"}
    assert any(c["label"] in {"UNSUPPORTED", "RISKY_ABSOLUTE"} for c in claims)


# --------------------------------------------------
# Explicit contradictions
# --------------------------------------------------

def test_contradiction_reject():
    req = SafeRAGRequest(
        request_id="test_contradiction",
        generated_text="Insulin is never used for type 2 diabetes."
    )

    decision, claims, _ = run_saferag(req)

    assert decision == "REJECT"
    assert any(c["label"] == "REFUTED" for c in claims)


# --------------------------------------------------
# Mixed claims
# --------------------------------------------------

def test_mixed_claims_refuse():
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
    assert any(c["label"] == "VERIFIED" for c in claims)
    assert any(c["label"] != "VERIFIED" for c in claims)


# --------------------------------------------------
# Long inputs
# --------------------------------------------------

def test_long_input_bounded():
    long_text = "Metformin is first line treatment. " * 100

    req = SafeRAGRequest(
        request_id="test_long",
        generated_text=long_text
    )

    decision, claims, _ = run_saferag(req)

    assert decision in {"REFUSE", "ACCEPT"}
    assert isinstance(claims, list)


# --------------------------------------------------
# Gibberish input
# --------------------------------------------------

def test_gibberish_refuse():
    req = SafeRAGRequest(
        request_id="test_gibberish",
        generated_text="asdkjh qweoiu zxcmn qweqwe"
    )

    decision, claims, _ = run_saferag(req)

    assert decision == "REFUSE"
    assert claims == []


# --------------------------------------------------
# Empty input
# --------------------------------------------------

def test_empty_input_refuse():
    req = SafeRAGRequest(
        request_id="test_empty",
        generated_text=""
    )

    decision, claims, _ = run_saferag(req)

    assert decision == "REFUSE"
    assert claims == []


# --------------------------------------------------
# Determinism
# --------------------------------------------------

def test_deterministic_execution():
    req = SafeRAGRequest(
        request_id="test_determinism",
        generated_text="Metformin is first line treatment."
    )

    out1 = run_saferag(req)
    out2 = run_saferag(req)

    assert out1 == out2


# --------------------------------------------------
# Audit logging
# --------------------------------------------------

def test_audit_log_written():
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
