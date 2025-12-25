from fastapi import FastAPI, HTTPException
from app.schemas import SafeRAGRequest, SafeRAGResponse
from app.service import run_saferag

app = FastAPI(title="SafeRAG Verification Service")


@app.get("/")
def root():
    return {
        "service": "SafeRAG",
        "status": "running",
        "docs": "/docs"
    }


@app.post("/verify", response_model=SafeRAGResponse)
def verify(req: SafeRAGRequest):
    decision, claims, metrics = run_saferag(req)

    if decision == "ERROR":
        raise HTTPException(
            status_code=500,
            detail="Internal SafeRAG error. See audit logs."
        )

    return {
        "decision": decision,
        "claims": claims,
        "metrics": metrics,
        "audit_id": req.request_id
    }
