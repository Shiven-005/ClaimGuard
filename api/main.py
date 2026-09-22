"""
FastAPI Application for ClaimGuard.
Exposes /health, /score, and /score_batch endpoints with Pydantic validation.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd

from claimguard.predict import get_predictor

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up predictor artifacts
    get_predictor()
    yield

app = FastAPI(
    title="ClaimGuard API",
    description="Insurance Claim Flagging, Tagging, and Risk Scoring API",
    version="0.1.0",
    lifespan=lifespan
)

class ClaimInputSchema(BaseModel):
    claim_id: Optional[str] = Field("CLM-000001", description="Unique Claim ID")
    policy_id: Optional[str] = Field("POL-000001", description="Policy ID")
    claimant_id: Optional[str] = Field("CLMNT-000001", description="Claimant ID")
    provider_id: Optional[str] = Field("PRV-0001", description="Healthcare or repair provider ID")
    claim_date: str = Field("2024-08-01", description="Claim filing date (YYYY-MM-DD)")
    policy_start_date: str = Field("2024-01-01", description="Policy inception date (YYYY-MM-DD)")
    claim_type: str = Field("auto", description="Claim type: auto, health, or property")
    claim_amount: float = Field(3500.0, description="Claim amount in currency units")
    peer_median_amount: float = Field(3500.0, description="Peer median amount for claim type and procedure")
    phone: Optional[str] = Field("555-019-2831", description="Claimant phone number")
    address: Optional[str] = Field("100 Main St, City", description="Claimant street address")
    bank_account: Optional[str] = Field("ACCT-12345678", description="Bank account identifier")
    garage_or_hospital_id: Optional[str] = Field("FAC-0001", description="Facility/garage/hospital ID")
    narrative: str = Field("Vehicle sustained bumper damage during low speed collision.", description="Text narrative of claim")
    num_prior_claims: int = Field(0, description="Number of prior claims filed by claimant")
    injury_flag: int = Field(0, description="1 if injury reported, 0 otherwise")
    police_report_flag: int = Field(0, description="1 if police report filed, 0 otherwise")
    region: Optional[str] = Field("North", description="Geographic region")
    age_group: Optional[str] = Field("26-40", description="Claimant age group")

class TagEvidenceSchema(BaseModel):
    tag: str
    confidence: float
    evidence: str

class ShapFeatureSchema(BaseModel):
    feature: str
    value: float
    shap_contribution: float

class ClaimScoreResponse(BaseModel):
    claim_id: str
    risk_score: float
    raw_probability: float
    action: str
    tags: List[TagEvidenceSchema]
    top_shap_features: List[ShapFeatureSchema]

class BatchScoreResponse(BaseModel):
    total_claims: int
    flagged_claims: int
    flagged_ratio: float
    results: List[Dict[str, Any]]


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}

@app.post("/score", response_model=ClaimScoreResponse)
def score_claim(claim: ClaimInputSchema):
    """Score a single insurance claim."""
    try:
        predictor = get_predictor()
        res = predictor.predict_single(claim.model_dump())
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score_batch", response_model=BatchScoreResponse)
def score_batch_endpoint(claims: List[ClaimInputSchema]):
    """Score a batch of insurance claims submitted as a JSON list."""
    try:
        predictor = get_predictor()
        claims_data = [c.model_dump() for c in claims]
        df_claims = pd.DataFrame(claims_data)
        df_res = predictor.predict_batch(df_claims)
        
        results = df_res.to_dict(orient="records")
        flagged_count = int((df_res['action'] != 'AUTO_APPROVE').sum())
        
        return {
            "total_claims": len(df_claims),
            "flagged_claims": flagged_count,
            "flagged_ratio": round(flagged_count / max(len(df_claims), 1), 4),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
