from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_score(claim, evidence):
    """
    Compute semantic similarity between claim and evidence.
    """
    embeddings = _model.encode([claim, evidence])
    return float(cosine_similarity(
        [embeddings[0]], [embeddings[1]]
    )[0][0])
