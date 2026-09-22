"""
Tests for synthetic data generation and reproducibility.
"""

import pandas as pd
from claimguard.synth import generate_synthetic_claims, get_time_splits

def test_generate_synthetic_claims_schema():
    df = generate_synthetic_claims(100, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    
    required_cols = [
        'claim_id', 'policy_id', 'claimant_id', 'provider_id', 'claim_date',
        'policy_start_date', 'claim_type', 'claim_amount', 'peer_median_amount',
        'phone', 'address', 'bank_account', 'garage_or_hospital_id', 'narrative',
        'num_prior_claims', 'injury_flag', 'police_report_flag', 'region', 'age_group',
        'is_fraud'
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col}"

def test_reproducibility():
    df1 = generate_synthetic_claims(50, seed=42)
    df2 = generate_synthetic_claims(50, seed=42)
    pd.testing.assert_frame_equal(df1, df2)

def test_time_splits():
    df = generate_synthetic_claims(500, seed=42)
    df_train, df_val, df_test, df_novel = get_time_splits(df)
    assert len(df_train) > 0
    assert len(df_val) > 0
    assert len(df_test) > 0
