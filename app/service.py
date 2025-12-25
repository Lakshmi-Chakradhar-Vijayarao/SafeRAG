from core.claims import extract_claims
from core.retriever import retrieve_evidence
from core.verifier import classify_claim, final_decision, load_policy
from core.metrics import compute_metrics
from app.audit import log_audit_event
from saferag_bootstrap import bootstrap


def run_saferag(request):
    """
    Production-grade SafeRAG execution.

    Guarantees:
    - deterministic
    - fail-safe
    - auditable
    - bounded
    """

    # --- ENSURE SYSTEM INITIALIZATION ---
    bootstrap()

    try:
        policy = load_policy(request.policy_profile)

        claims = extract_claims(
            request.generated_text,
            mode=policy.get("claim_extraction_mode", "strict"),
            max_claims=policy.get("max_claims", 10)
        )

        # Explicit empty-claim behavior
        if not claims:
            decision = policy.get("on_insufficient", "REFUSE")
            claim_results = []
            metrics = {}

            log_audit_event({
                "audit_id": request.request_id,
                "domain": request.domain,
                "input": {
                    "generated_text": request.generated_text,
                    "policy_profile": request.policy_profile
                },
                "claims": claim_results,
                "metrics": metrics,
                "decision": decision,
                "note": "No extractable claims"
            })

            return decision, claim_results, metrics

        claim_results = []

        for claim in claims:
            evidences = retrieve_evidence(
                claim,
                top_k=policy.get("max_evidence_per_claim", 3)
            )

            verdicts = [
                classify_claim(claim, ev["text"])
                for ev in evidences
            ]

            # Conservative aggregation
            if any(v["label"] == "CONTRADICTED" for v in verdicts):
                final_label = "CONTRADICTED"
            elif any(v["label"] == "SUPPORTED" for v in verdicts):
                final_label = "SUPPORTED"
            else:
                final_label = "INSUFFICIENT_EVIDENCE"

            best = max(verdicts, key=lambda v: v["semantic_score"])

            claim_results.append({
                "claim": claim,
                "label": final_label,
                "score": float(best["semantic_score"]),
                "evidence_ids": [str(i) for i in range(len(evidences))]
            })

        metrics = compute_metrics(
            [
                {"label": c["label"], "semantic_score": c["score"]}
                for c in claim_results
            ]
        )

        decision = final_decision(
            [{"label": c["label"]} for c in claim_results],
            request.policy_profile
        )

    except Exception as e:
        # --- SYSTEM ERROR (NOT A SAFETY REFUSE) ---
        decision = "ERROR"
        claim_results = []
        metrics = {}

        log_audit_event({
            "audit_id": request.request_id,
            "domain": request.domain,
            "input": {
                "generated_text": request.generated_text,
                "policy_profile": request.policy_profile
            },
            "claims": claim_results,
            "metrics": metrics,
            "decision": decision,
            "error": str(e)
        })

        return decision, claim_results, metrics

    # --- NORMAL AUDIT LOGGING ---
    log_audit_event({
        "audit_id": request.request_id,
        "domain": request.domain,
        "input": {
            "generated_text": request.generated_text,
            "policy_profile": request.policy_profile
        },
        "claims": claim_results,
        "metrics": metrics,
        "decision": decision
    })

    return decision, claim_results, metrics
