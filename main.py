from retriever import EvidenceRetriever
from claims import extract_claims
from verifier import classify_claim
from metrics import compute_metrics


def load_lines(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def main():
    documents = load_lines("data/documents.txt")
    generations = load_lines("data/generations.txt")

    retriever = EvidenceRetriever(documents)
    generation = generations[0]

    print("\nMODEL OUTPUT:\n", generation)

    claims = extract_claims(generation)
    verification_results = []

    print("\nCLAIM VERIFICATION:")

    for claim in claims:
        evidence_list = retriever.retrieve(claim, top_k=3)

        votes = []
        for ev in evidence_list:
            result = classify_claim(claim, ev["text"])
            result["evidence_id"] = ev["doc_id"]
            result["evidence_score"] = ev["score"]
            votes.append(result)

        labels = [v["label"] for v in votes]
        if "CONTRADICTED" in labels:
            final_label = "CONTRADICTED"
        elif "SUPPORTED" in labels:
            final_label = "SUPPORTED"
        else:
            final_label = "INSUFFICIENT_EVIDENCE"

        final = votes[0]
        final["label"] = final_label
        verification_results.append(final)

        print(f"\nClaim: {claim}")
        print("Final Label:", final_label)
        print("Evidence:", final["evidence"])
        print("Semantic Score:", final["semantic_score"])

    metrics = compute_metrics(verification_results)

    print("\nMETRICS:", metrics)

    # === SAFETY DECISION ===
    if metrics["contradiction_rate"] > 0:
        decision = "REJECT"
    elif metrics["insufficient_rate"] > 0.3:
        decision = "REFUSE"
    else:
        decision = "ACCEPT"

    print("\nSYSTEM DECISION:", decision)


if __name__ == "__main__":
    main()