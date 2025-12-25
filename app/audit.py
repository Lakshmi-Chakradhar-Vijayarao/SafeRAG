import json
import time
from pathlib import Path

LOG_PATH = Path("logs/saferag_audit.jsonl")


def log_audit_event(payload: dict):
    """
    Append a single audit event as JSONL.
    Must NEVER crash the main system.
    """
    try:
        payload["timestamp"] = time.time()

        # Ensure log directory exists
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(payload) + "\n")

    except Exception:
        # Safety system must not fail due to logging
        pass
