import re


# Minimal linguistic markers of factual propositions
CLAIM_VERBS = {
    "is", "are", "was", "were",
    "should", "must", "can", "will",
    "has", "have", "had",
    "not", "never"
}


def extract_claims(text, mode="strict", max_claims=10):
    """
    Extract atomic factual claims from generated text.

    A claim must:
    - Contain alphabetic words
    - Contain minimal propositional structure (verbs/modals)
    - Be bounded and deterministic

    Modes:
    - strict: empty claims => REFUSE
    - fallback: entire text becomes a claim if none extracted
    """

    if not text or len(text.strip()) < 5:
        return []

    sentences = re.split(r"[.?!]", text)
    claims = []

    for s in sentences:
        s = s.strip()
        if len(s) < 5:
            continue

        tokens = s.lower().split()

        # Reject low-signal text (no verbs / propositions)
        if not any(tok in CLAIM_VERBS for tok in tokens):
            continue

        # Reject non-linguistic strings
        if not re.search(r"[a-zA-Z]{3,}", s):
            continue

        parts = re.split(r"\band\b|\bbut\b", s)
        for p in parts:
            p = p.strip()
            ptokens = p.lower().split()

            if (
                len(p) > 5
                and any(tok in CLAIM_VERBS for tok in ptokens)
                and re.search(r"[a-zA-Z]{3,}", p)
            ):
                claims.append(p)

            if len(claims) >= max_claims:
                break

        if len(claims) >= max_claims:
            break

    if not claims and mode == "fallback":
        return [text.strip()]

    return claims[:max_claims]
