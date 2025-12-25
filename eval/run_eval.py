import sys
import json
from pathlib import Path

# --------------------------------------------------
# Ensure project root is on PYTHONPATH
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from saferag_bootstrap import bootstrap
from app.service import run_saferag
from app.schemas import SafeRAGRequest


# --------------------------------------------------
# Baseline: Pass-through (NO gating)
# --------------------------------------------------
def run_pass_through_baseline(path):
    """
    Baseline evaluation:
    - Runs the full pipeline
    - Ignores ACCEPT / REFUSE / REJECT
    - Counts how many claims are NOT VERIFIED
    - Represents raw hallucination generation rate
    """

    total_claims = 0
    hallucinated_claims = 0

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue

            ex = json.loads(line)

            req = SafeRAGRequest(
                request_id=ex["id"],
                generated_text=ex["generation"],
            )

            _, claims, _ = run_saferag(req)

            total_claims += len(claims)
            hallucinated_claims += sum(
                c["label"] != "VERIFIED" for c in claims
            )

    print("Baseline (no gating):")
    print("  Total claims:", total_claims)
    print("  Hallucinated claims:", hallucinated_claims)
    print(
        "  Hallucination rate:",
        round(hallucinated_claims / max(total_claims, 1), 3),
    )


# --------------------------------------------------
# SafeRAG Evaluation: Safety-gated exposure
# --------------------------------------------------
def run_eval(path):
    """
    SafeRAG evaluation:
    - Hallucinations count ONLY if decision == ACCEPT
    - REFUSE / REJECT are treated as successful safety containment
    - Uses final truth labels:
        VERIFIED | REFUTED | UNSUPPORTED | RISKY_ABSOLUTE
    """

    total_claims = 0
    passed_hallucinations = 0

    blocked_claims = 0
    safe_refusals = 0
    hard_rejections = 0

    decision_counts = {
        "ACCEPT": 0,
        "REFUSE": 0,
        "REJECT": 0,
    }

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue

            ex = json.loads(line)

            req = SafeRAGRequest(
                request_id=ex["id"],
                generated_text=ex["generation"],
            )

            decision, claims, _ = run_saferag(req)

            decision_counts[decision] += 1
            total_claims += len(claims)

            # --------------------------------------
            # Exposure vs containment
            # --------------------------------------
            if decision == "ACCEPT":
                passed_hallucinations += sum(
                    c["label"] in {
                        "REFUTED",
                        "UNSUPPORTED",
                        "RISKY_ABSOLUTE",
                    }
                    for c in claims
                )
            else:
                blocked_claims += len(claims)

                if decision == "REFUSE":
                    safe_refusals += 1
                elif decision == "REJECT":
                    hard_rejections += 1

    print("SafeRAG (with gating):")
    print("  Total claims:", total_claims)
    print("  Hallucinations reaching user:", passed_hallucinations)
    print(
        "  Pass-through hallucination rate:",
        round(passed_hallucinations / max(total_claims, 1), 3),
    )

    print("\nSafety containment:")
    print("  Blocked claims:", blocked_claims)
    print("  Safe refusals (uncertainty):", safe_refusals)
    print("  Hard rejections (contradictions):", hard_rejections)

    print("\nDecision distribution:")
    for k in ["ACCEPT", "REFUSE", "REJECT"]:
        print(f"  {k}: {decision_counts[k]}")


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    bootstrap()

    print("=== CLINICAL DATASET ===")
    run_pass_through_baseline("eval/datasets/clinical.jsonl")
    run_eval("eval/datasets/clinical.jsonl")

    print("\n=== FINANCE DATASET ===")
    run_pass_through_baseline("eval/datasets/finance.jsonl")
    run_eval("eval/datasets/finance.jsonl")
