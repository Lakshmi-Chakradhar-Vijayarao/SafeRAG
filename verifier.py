from semantic import semantic_score

NEGATION_TERMS = {"not", "no", "never", "avoid", "contraindicated"}

def classify_claim(claim, evidence):
    """
    Classify a claim against evidence using semantic + lexical signals.
    """

    claim_tokens = set(claim.lower().split())
    evidence_tokens = set(evidence.lower().split())

    lexical_overlap = len(claim_tokens & evidence_tokens) / max(len(claim_tokens), 1)
    semantic = semantic_score(claim, evidence)

    neg_claim = any(t in claim_tokens for t in NEGATION_TERMS)
    neg_evidence = any(t in evidence_tokens for t in NEGATION_TERMS)

    if semantic >= 0.65:
        if neg_claim != neg_evidence:
            label = "CONTRADICTED"
        else:
            label = "SUPPORTED"
    elif semantic >= 0.4:
        label = "INSUFFICIENT_EVIDENCE"
    else:
        label = "INSUFFICIENT_EVIDENCE"

    return {
        "claim": claim,
        "label": label,
        "semantic_score": round(semantic, 3),
        "lexical_overlap": round(lexical_overlap, 3),
        "evidence": evidence
    }
