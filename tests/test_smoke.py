"""
Smoke test asserting that an injected fraud claim scores higher than a clean claim.
"""

from claimguard.predict import score_single

def test_fraud_vs_clean_smoke_test():
    clean_claim = {
        'claim_id': 'CLM-CLEAN-001',
        'policy_id': 'POL-10001',
        'claimant_id': 'CLMNT-10001',
        'provider_id': 'PRV-1001',
        'claim_date': '2024-08-01',
        'policy_start_date': '2023-08-01',
        'claim_type': 'auto',
        'claim_amount': 3200.0,
        'peer_median_amount': 3500.0,
        'phone': '555-111-0001',
        'address': '100 Main St',
        'bank_account': 'ACCT-0001',
        'garage_or_hospital_id': 'FAC-1001',
        'narrative': 'Minor parking lot scratch on door. Normal handling.',
        'num_prior_claims': 0,
        'injury_flag': 0,
        'police_report_flag': 0,
        'region': 'North',
        'age_group': '26-40'
    }

    injected_fraud_claim = {
        'claim_id': 'CLM-FRAUD-001',
        'policy_id': 'POL-99999',
        'claimant_id': 'CLMNT-99999',
        'provider_id': 'PRV-1495', # Malicious provider
        'claim_date': '2024-08-01',
        'policy_start_date': '2024-07-25', # Suspicious timing: 7 days after policy start!
        'claim_type': 'auto',
        'claim_amount': 16500.0, # Inflated amount: 4.7x peer median!
        'peer_median_amount': 3500.0,
        'phone': '555-999-0001',
        'address': '100 Main St',
        'bank_account': 'ACCT-9999',
        'garage_or_hospital_id': 'FAC-1001',
        'narrative': 'No injuries sustained during collision.', # Narrative inconsistency (injury_flag=1)
        'num_prior_claims': 3,
        'injury_flag': 1,
        'police_report_flag': 0,
        'region': 'North',
        'age_group': '26-40'
    }

    res_clean = score_single(clean_claim)
    res_fraud = score_single(injected_fraud_claim)

    print(f"Clean Risk Score: {res_clean['risk_score']}, Fraud Risk Score: {res_fraud['risk_score']}")
    assert res_fraud['risk_score'] > res_clean['risk_score']
    assert res_fraud['risk_score'] >= 50.0
    assert res_clean['risk_score'] < 50.0
    assert res_fraud['action'] in ['REVIEW', 'ESCALATE']
    assert len(res_fraud['tags']) > 0
