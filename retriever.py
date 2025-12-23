from rank_bm25 import BM25Okapi


class EvidenceRetriever:
    """
    Evidence retriever for SafeRAG.
    Returns evidence passages with document IDs and BM25 scores.
    """

    def __init__(self, documents):
        self.documents = documents
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, claim, top_k=3):
        tokens = claim.lower().split()
        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(self.documents)),
            key=lambda i: scores[i],
            reverse=True
        )

        return [
            {
                "doc_id": idx,
                "text": self.documents[idx],
                "score": round(scores[idx], 3)
            }
            for idx in ranked_indices[:top_k]
        ]
