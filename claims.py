import re

def extract_claims(text):
    """
    Extract atomic factual claims.
    Splits on punctuation and conjunctions.
    """
    sentences = re.split(r"[.?!]", text)
    claims = []

    for s in sentences:
        s = s.strip()
        if len(s) < 5:
            continue

        # Split compound claims
        parts = re.split(r"\band\b|\bbut\b", s)
        for p in parts:
            p = p.strip()
            if len(p) > 5:
                claims.append(p)

    return claims
