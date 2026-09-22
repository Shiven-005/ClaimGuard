"""
Tests for single and batch claim prediction schema.
"""

from claimguard.predict import score_single, score_batch
from claimguard.synth import generate_synthetic_claims

def test_score_single_schema():
    claim = {
        'claim_id': 'CLM-TEST-001',
        'policy_id': 'POL-10001',
        'claimant_id': 'CLMNT-10001',
        'provider_id': 'PRV-1001',
        'claim_date': '2024-08-01',
        'policy_start_date': '2022-01-01',
        'claim_type': 'auto',
        'claim_amount': 3500.0,
        'peer_median_amount': 3500.0,
        'phone': '555-000-1111',
        'address': '123 Main St',
        'bank_account': 'ACCT-1111',
        'garage_or_hospital_id': 'FAC-1001',
        'narrative': 'Low speed collision.',
        'num_prior_claims': 0,
        'injury_flag': 0,
        'police_report_flag': 0,
        'region': 'North',
        'age_group': '26-40'
    }

    res = score_single(claim)
    assert 'risk_score' in res
    assert 0.0 <= res['risk_score'] <= 100.0
    assert res['action'] in ['AUTO_APPROVE', 'REVIEW', 'ESCALATE']
    assert isinstance(res['tags'], list)
    assert isinstance(res['top_shap_features'], list)
    assert len(res['top_shap_features']) <= 5

def test_score_batch():
    df = generate_synthetic_claims(20, seed=42)
    res_df = score_batch(df)
    assert 'risk_score' in res_df.columns
    assert 'action' in res_df.columns
    assert 'flagged_tags' in res_df.columns
    assert len(res_df) == 20
