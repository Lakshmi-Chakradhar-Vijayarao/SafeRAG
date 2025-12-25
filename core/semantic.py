from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _model


def semantic_score(claim, evidence):
    try:
        model = _get_model()
        embeddings = model.encode(
            [claim, evidence],
            batch_size=2,
            show_progress_bar=False
        )
        return float(
            cosine_similarity(
                [embeddings[0]], [embeddings[1]]
            )[0][0]
        )
    except Exception:
        # Fail-safe semantic fallback
        return 0.0
