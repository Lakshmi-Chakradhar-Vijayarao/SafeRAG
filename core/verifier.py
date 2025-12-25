from pathlib import Path
import yaml
from core.semantic import semantic_score

# -------------------------
# Constants
# -------------------------

NEGATION_TERMS = {"not", "no", "never", "avoid", "contraindicated"}

DEFAULT_POLICY = {
    "contradiction_threshold": 0.7,
    "min_support_rate": 0.6,
    "on_insufficient": "REFUSE"
}


# -------------------------
# Claim Classification
# -------------------------

def classify_claim(claim, evidence):
    """
    Classify a claim against a single evidence passage.

    Labels:
    - SUPPORTED
    - CONTRADICTED
    - INSUFFICIENT_EVIDENCE

    Design:
    - Conservative
    - Safety-first
    - Deterministic
    """

    claim_tokens = set(claim.lower().split())
    evidence_tokens = set(evidence.lower().split())

    semantic = semantic_score(claim, evidence)

    neg_claim = any(t in claim_tokens for t in NEGATION_TERMS)
    neg_evidence = any(t in evidence_tokens for t in NEGATION_TERMS)

    # -------------------------
    # Explicit contradiction rule
    # -------------------------
    # Strong negation statements should be treated as contradictions
    # even if retrieval evidence is imperfect.
    if neg_claim and semantic >= 0.4:
        label = "CONTRADICTED"

    # -------------------------
    # Standard semantic logic
    # -------------------------
    elif semantic >= 0.65:
        label = "CONTRADICTED" if neg_claim != neg_evidence else "SUPPORTED"
    else:
        label = "INSUFFICIENT_EVIDENCE"

    return {
        "claim": claim,
        "label": label,
        "semantic_score": round(float(semantic), 3),
        "lexical_overlap": round(
            len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1),
            3
        ),
        "evidence": evidence
    }


# -------------------------
# Policy Loading
# -------------------------

def load_policy(profile="default"):
    """
    Load verification policy from YAML.
    Falls back to DEFAULT_POLICY if missing.
    """
    path = Path("policies") / f"{profile}.yaml"
    if not path.exists():
        return DEFAULT_POLICY
    return yaml.safe_load(path.read_text())


# -------------------------
# System-Level Decision
# -------------------------

def final_decision(claim_results, policy_profile="default"):
    """
    Enforce system-level safety decision.

    Returns:
    - ACCEPT
    - REJECT
    - REFUSE
    """

    policy = load_policy(policy_profile)

    # Immediate rejection on contradiction
    if any(c["label"] == "CONTRADICTED" for c in claim_results):
        return "REJECT"

    support_rate = (
        sum(c["label"] == "SUPPORTED" for c in claim_results)
        / max(len(claim_results), 1)
    )

    if support_rate < policy["min_support_rate"]:
        return policy["on_insufficient"]

    return "ACCEPT"
