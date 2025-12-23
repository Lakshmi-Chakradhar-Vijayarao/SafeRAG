def compute_metrics(results):
    total = len(results)
    if total == 0:
        return {}

    supported = sum(r["label"] == "SUPPORTED" for r in results)
    contradicted = sum(r["label"] == "CONTRADICTED" for r in results)
    insufficient = sum(r["label"] == "INSUFFICIENT_EVIDENCE" for r in results)

    avg_semantic = sum(r.get("semantic_score", 0) for r in results) / total

    return {
        "support_rate": round(supported / total, 3),
        "contradiction_rate": round(contradicted / total, 3),
        "insufficient_rate": round(insufficient / total, 3),
        "hallucination_rate": round((contradicted + insufficient) / total, 3),
        "avg_semantic_score": round(avg_semantic, 3)
    }
