from pathlib import Path
import yaml

DEFAULT_POLICY = {
    "min_support_rate": 0.6,
    "on_insufficient": "REFUSE",
    "claim_extraction_mode": "strict",
    "max_claims": 10,
    "max_evidence_per_claim": 3,
}


def load_policy(profile="default"):
    path = Path("policies") / f"{profile}.yaml"
    if not path.exists():
        return DEFAULT_POLICY

    policy = yaml.safe_load(path.read_text())
    return {**DEFAULT_POLICY, **policy}
