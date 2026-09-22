"""
Tests for FastAPI endpoints via TestClient.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "0.1.0"}

def test_score_endpoint():
    claim_payload = {
        "claim_id": "CLM-API-001",
        "policy_id": "POL-10001",
        "claimant_id": "CLMNT-10001",
        "provider_id": "PRV-1001",
        "claim_date": "2024-08-01",
        "policy_start_date": "2022-01-01",
        "claim_type": "auto",
        "claim_amount": 3500.0,
        "peer_median_amount": 3500.0,
        "phone": "555-111-2222",
        "address": "456 Oak St",
        "bank_account": "ACCT-2222",
        "garage_or_hospital_id": "FAC-1001",
        "narrative": "Fender bender while parking.",
        "num_prior_claims": 0,
        "injury_flag": 0,
        "police_report_flag": 0,
        "region": "North",
        "age_group": "26-40"
    }

    response = client.post("/score", json=claim_payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "action" in data
    assert "tags" in data
    assert "top_shap_features" in data

def test_score_batch_endpoint():
    claims_payload = [
        {
            "claim_id": f"CLM-API-{i}",
            "policy_id": "POL-10001",
            "claimant_id": "CLMNT-10001",
            "provider_id": "PRV-1001",
            "claim_date": "2024-08-01",
            "policy_start_date": "2022-01-01",
            "claim_type": "auto",
            "claim_amount": 3500.0,
            "peer_median_amount": 3500.0,
            "phone": "555-111-2222",
            "address": "456 Oak St",
            "bank_account": "ACCT-2222",
            "garage_or_hospital_id": "FAC-1001",
            "narrative": "Fender bender.",
            "num_prior_claims": 0,
            "injury_flag": 0,
            "police_report_flag": 0,
            "region": "North",
            "age_group": "26-40"
        }
        for i in range(3)
    ]

    response = client.post("/score_batch", json=claims_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_claims"] == 3
    assert len(data["results"]) == 3
