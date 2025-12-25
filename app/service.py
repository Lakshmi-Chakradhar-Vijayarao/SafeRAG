"""
SafeRAG execution service.

RESPONSIBILITIES:
- Orchestrate claim extraction
- Retrieve evidence
- Aggregate claim truth states
- Apply policy to reach system decision
- Emit metrics and audit logs

NOTE:
- Verifier returns ONLY truth labels
- ACCEPT is allowed ONLY if all claims are VERIFIED
- Any REFUTED claim blocks ACCEPT (global safety rule)
"""

from saferag_bootstrap import bootstrap
from core.claims import extract_claims
from core.retriever import retrieve_evidence
from core.verifier import classify_claim
from app.audit import log_audit_event
from core.policy import load_policy


# --------------------------------------------------
# Lexical similarity (deterministic, no embeddings)
# --------------------------------------------------

def token_overlap_ratio(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# --------------------------------------------------
# Claim clustering (for analysis / metrics only)
# --------------------------------------------------

def cluster_claims(claim_results, threshold=0.5):
    clusters = []

    for c in claim_results:
        placed = False
        for cluster in clusters:
            if token_overlap_ratio(c["claim"], cluster[0]["claim"]) >= threshold:
                cluster.append(c)
                placed = True
                break
        if not placed:
            clusters.append([c])

    clusters.sort(key=len, reverse=True)
    return clusters


# --------------------------------------------------
# Main execution
# --------------------------------------------------

def run_saferag(request):
    """
    Execute SafeRAG end-to-end.

    Returns:
        decision: ACCEPT | REFUSE | REJECT
        claim_results: list
        metrics: dict
    """

    bootstrap()

    try:
        policy = load_policy(request.policy_profile)

        # --------------------------------------------------
        # Claim extraction
        # --------------------------------------------------
        claims = extract_claims(
            request.generated_text,
            mode=policy.get("claim_extraction_mode", "strict"),
            max_claims=policy.get("max_claims", 10),
        )

        if not claims:
            decision = policy.get("on_insufficient", "REFUSE")
            log_audit_event({
                "audit_id": request.request_id,
                "decision": decision,
                "claims": [],
            })
            return decision, [], {}

        # --------------------------------------------------
        # Claim verification
        # --------------------------------------------------
        claim_results = []

        for claim in claims:
            evidences = retrieve_evidence(
                claim,
                top_k=policy.get("max_evidence_per_claim", 3),
            )

            verdicts = [
                classify_claim(claim, ev["text"])
                for ev in evidences
            ]

            labels = [v["label"] for v in verdicts]

            # Claim-level priority
            if "REFUTED" in labels:
                final = next(v for v in verdicts if v["label"] == "REFUTED")
            elif "VERIFIED" in labels:
                final = next(v for v in verdicts if v["label"] == "VERIFIED")
            elif "RISKY_ABSOLUTE" in labels:
                final = next(v for v in verdicts if v["label"] == "RISKY_ABSOLUTE")
            else:
                final = verdicts[0]  # UNSUPPORTED

            claim_results.append({
                "claim": claim,
                "label": final["label"],
                "semantic_score": final["semantic_score"],
            })

        # --------------------------------------------------
        # Metrics (dominant cluster for reporting only)
        # --------------------------------------------------
        clusters = cluster_claims(claim_results)
        dominant_cluster = clusters[0]

        dominant_labels = [c["label"] for c in dominant_cluster]
        verified = dominant_labels.count("VERIFIED")
        refuted = dominant_labels.count("REFUTED")
        total = len(dominant_labels)

        metrics = {
            "support_rate": round(verified / max(total, 1), 3),
            "contradiction_rate": round(refuted / max(total, 1), 3),
        }

        # --------------------------------------------------
        # SYSTEM-LEVEL DECISION (SAFETY-FIRST)
        # --------------------------------------------------
        all_labels = [c["label"] for c in claim_results]

        # Absolute safety rule
        if "REFUTED" in all_labels:
            decision = "REJECT"

        # Accept only if EVERY claim is verified
        elif all(l == "VERIFIED" for l in all_labels):
            decision = "ACCEPT"

        # Everything else → uncertainty
        else:
            decision = policy.get("on_insufficient", "REFUSE")

    except Exception as e:
        log_audit_event({
            "audit_id": request.request_id,
            "decision": "ERROR",
            "error": str(e),
        })
        return "ERROR", [], {}

    # --------------------------------------------------
    # Audit log
    # --------------------------------------------------
    log_audit_event({
        "audit_id": request.request_id,
        "decision": decision,
        "claims": claim_results,
        "metrics": metrics,
    })

    return decision, claim_results, metrics
