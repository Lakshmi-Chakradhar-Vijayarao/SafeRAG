from pydantic import BaseModel
from typing import List, Dict
from enum import Enum


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REFUSE = "REFUSE"


class SafeRAGRequest(BaseModel):
    request_id: str
    generated_text: str
    domain: str = "default"
    policy_profile: str = "default"


class ClaimResult(BaseModel):
    claim: str
    label: str
    score: float
    evidence_ids: List[str]


class SafeRAGResponse(BaseModel):
    decision: Decision
    claims: List[ClaimResult]
    metrics: Dict[str, float]
    audit_id: str
