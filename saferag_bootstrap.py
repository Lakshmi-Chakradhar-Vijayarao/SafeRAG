import multiprocessing as mp
from pathlib import Path
from core.retriever import initialize_retriever

_BOOTSTRAPPED = False


def bootstrap():
    """
    Initialize all global SafeRAG components.

    This function is:
    - idempotent
    - safe to call multiple times
    - REQUIRED before any verification
    """
    global _BOOTSTRAPPED

    if _BOOTSTRAPPED:
        return

    # Required for macOS / PyTorch safety
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    docs_path = Path("data/documents.txt")
    if not docs_path.exists():
        raise RuntimeError("Missing data/documents.txt")

    documents = [
        line.strip()
        for line in docs_path.read_text().splitlines()
        if line.strip()
    ]

    initialize_retriever(documents)
    _BOOTSTRAPPED = True
