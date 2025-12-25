import os
from sklearn.metrics.pairwise import cosine_similarity

_model = None


def _get_model():
    global _model
    if _model is None:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    return _model


def semantic_score(claim: str, evidence: str) -> float:
    """
    Realistic semantic scoring with CI-safe fallback.
    """

    # Fast / test mode
    if os.environ.get("SAFERAG_NO_EMBEDDINGS") == "1":
        c = claim.lower()
        e = evidence.lower()

        if c in e or e in c:
            return 0.75

        overlap = len(set(c.split()) & set(e.split()))
        if overlap >= 2:
            return 0.45

        return 0.1  # <-- CRITICAL realism: non-zero noise

    try:
        model = _get_model()
        emb = model.encode([claim, evidence], convert_to_numpy=True)
        return float(cosine_similarity([emb[0]], [emb[1]])[0][0])
    except Exception:
        return 0.1
