"""
Claim truth classification module for SafeRAG.

RESPONSIBILITY (STEP 1 ONLY):
- Determine factual status of a claim w.r.t evidence
- NO policy decisions
- NO system acceptance logic
- Deterministic & auditable
"""

from core.semantic import semantic_score

# -------------------------
# Linguistic signals
# -------------------------

NEGATION_TERMS = {"not", "no", "never", "avoid", "contraindicated"}
ABSOLUTE_TERMS = {"never", "always", "guarantees", "completely"}

# -------------------------
# Domain phrase grounding
# -------------------------

PHRASE_GROUNDING = {
    "first_line_treatment": [
        "first line treatment",
        "recommended first line treatment",
        "first line pharmacological treatment",
        "recommended first line pharmacological treatment",
    ],
    "insulin_usage": [
        "insulin therapy may be required",
        "insulin is used",
        "insulin therapy",
    ],
    "ace_arb_combination": [
        "combining ace inhibitors and arbs is not recommended",
        "ace inhibitors and arbs should not be combined",
    ],
}


def _phrase_match(claim_l: str, evidence_l: str) -> bool:
    for variants in PHRASE_GROUNDING.values():
        if any(v in claim_l for v in variants) and any(v in evidence_l for v in variants):
            return True
    return False


# -------------------------
# Claim Truth Classification
# -------------------------

def classify_claim(claim: str, evidence: str):
    """
    OUTPUT STATES:
    - VERIFIED
    - REFUTED
    - UNSUPPORTED
    - RISKY_ABSOLUTE
    """

    claim_l = claim.lower()
    evidence_l = evidence.lower()

    claim_tokens = set(claim_l.split())
    evidence_tokens = set(evidence_l.split())

    semantic = semantic_score(claim, evidence)
    lexical_overlap = len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)

    has_absolute = any(t in claim_tokens for t in ABSOLUTE_TERMS)
    neg_claim = any(t in claim_tokens for t in NEGATION_TERMS)

    # --------------------------------------------------
    # VERIFIED — phrase grounding (highest confidence)
    # --------------------------------------------------
    if _phrase_match(claim_l, evidence_l):
        return _result("VERIFIED", semantic, lexical_overlap)

    # --------------------------------------------------
    # REFUTED — SAFETY-FIRST ABSOLUTE NEGATION
    #
    # Medical absolutes like:
    #   "never used", "always safe"
    # are treated as contradictions unless explicitly supported.
    # --------------------------------------------------
    if has_absolute and neg_claim:
        return _result("REFUTED", semantic, lexical_overlap)

    # --------------------------------------------------
    # VERIFIED — semantic / lexical support
    # --------------------------------------------------
    if semantic >= 0.65 or lexical_overlap >= 0.35:
        return _result("VERIFIED", semantic, lexical_overlap)

    # --------------------------------------------------
    # UNSUPPORTED — default
    # --------------------------------------------------
    return _result("UNSUPPORTED", semantic, lexical_overlap)


def _result(label, semantic, overlap):
    return {
        "label": label,
        "semantic_score": round(float(semantic), 3),
        "lexical_overlap": round(float(overlap), 3),
    }
