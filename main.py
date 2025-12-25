from core.retriever import initialize_retriever
from core.claims import extract_claims
from core.verifier import classify_claim, final_decision
from core.metrics import compute_metrics
from core.retriever import retrieve_evidence


def load_lines(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def main():
    documents = load_lines("data/documents.txt")
    generations = load_lines("data/generations.txt")

    initialize_retriever(documents)

    generation = generations[0]
    print("\nMODEL OUTPUT:\n", generation)

    claims = extract_claims(generation)
    results = []

    for claim in claims:
        evidence_list = retrieve_evidence(claim)

        verdicts = [classify_claim(claim, ev["text"]) for ev in evidence_list]

        if any(v["label"] == "CONTRADICTED" for v in verdicts):
            label = "CONTRADICTED"
        elif any(v["label"] == "SUPPORTED" for v in verdicts):
            label = "SUPPORTED"
        else:
            label = "INSUFFICIENT_EVIDENCE"

        best = max(verdicts, key=lambda v: v["semantic_score"])
        best["label"] = label
        results.append(best)

        print(f"\nClaim: {claim}")
        print("Final Label:", label)
        print("Evidence:", best["evidence"])
        print("Semantic Score:", best["semantic_score"])

    metrics = compute_metrics(results)
    decision = final_decision(results)

    print("\nMETRICS:", metrics)
    print("\nSYSTEM DECISION:", decision)


if __name__ == "__main__":
    main()
