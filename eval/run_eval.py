import sys
from pathlib import Path

# --------------------------------------------------
# Ensure project root is on PYTHONPATH
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import json
from saferag_bootstrap import bootstrap
from app.service import run_saferag
from app.schemas import SafeRAGRequest


# --------------------------------------------------
# Baseline: No gating (hallucinations observed only)
# --------------------------------------------------
def run_pass_through_baseline(path):
    """
    Baseline evaluation:
    - Claims are generated and verified
    - No safety gating is applied
    - Measures how many hallucinations exist
    """

    total_claims = 0
    hallucinated_claims = 0

    with open(path) as f:
        for line in f:
            ex = json.loads(line)

            req = SafeRAGRequest(
                request_id=ex["id"],
                generated_text=ex["generation"]
            )

            # Run verification but DO NOT gate
            _, claims, _ = run_saferag(req)

            total_claims += len(claims)
            hallucinated_claims += sum(
                c["label"] != "SUPPORTED" for c in claims
            )

    print("Baseline (no gating):")
    print("Total claims:", total_claims)
    print("Hallucinated claims:", hallucinated_claims)
    print(
        "Hallucination rate:",
        round(hallucinated_claims / max(total_claims, 1), 3)
    )


# --------------------------------------------------
# SafeRAG Evaluation: Measure pass-through hallucinations
# --------------------------------------------------
def run_eval(path):
    """
    SafeRAG evaluation:
    - Measures hallucinations that actually reach the user
    - Hallucinations are counted ONLY if decision == ACCEPT
    """

    total_claims = 0
    passed_hallucinations = 0

    with open(path) as f:
        for line in f:
            ex = json.loads(line)

            req = SafeRAGRequest(
                request_id=ex["id"],
                generated_text=ex["generation"]
            )

            decision, claims, _ = run_saferag(req)
            total_claims += len(claims)

            # Only count hallucinations that pass through
            if decision == "ACCEPT":
                passed_hallucinations += sum(
                    c["label"] != "SUPPORTED" for c in claims
                )

    print("SafeRAG (with gating):")
    print("Total claims:", total_claims)
    print("Hallucinations reaching user:", passed_hallucinations)
    print(
        "Pass-through hallucination rate:",
        round(passed_hallucinations / max(total_claims, 1), 3)
    )


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    # Initialize SafeRAG components
    bootstrap()

    print("=== CLINICAL DATASET ===")
    run_pass_through_baseline("eval/datasets/clinical.jsonl")
    run_eval("eval/datasets/clinical.jsonl")

    print("\n=== FINANCE DATASET ===")
    run_pass_through_baseline("eval/datasets/finance.jsonl")
    run_eval("eval/datasets/finance.jsonl")
