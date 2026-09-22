"""
Synthetic Data Generator for ClaimGuard.
Generates 50,000 claims over 3 years with time-based splits and ground-truth fraud tags.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_claims(n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)

    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    total_days = (end_date - start_date).days

    # Base IDs
    claim_ids = [f"CLM-{i+100000:06d}" for i in range(n_samples)]
    policy_ids = [f"POL-{random.randint(10000, 45000):06d}" for _ in range(n_samples)]
    claimant_ids = [f"CLMNT-{random.randint(10000, 30000):06d}" for _ in range(n_samples)]
    provider_ids = [f"PRV-{random.randint(1000, 1500):04d}" for _ in range(n_samples)]
    
    # Claim dates distributed uniformly over 3 years
    random_days = np.random.randint(0, total_days, size=n_samples)
    claim_dates = [start_date + timedelta(days=int(d)) for d in random_days]

    # Policy start dates (typically 30 to 730 days prior)
    policy_lags = np.random.randint(30, 730, size=n_samples)
    policy_start_dates = [c_date - timedelta(days=int(lag)) for c_date, lag in zip(claim_dates, policy_lags)]

    # Categoricals
    claim_types = np.random.choice(['auto', 'health', 'property'], size=n_samples, p=[0.45, 0.35, 0.20])
    regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], size=n_samples, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    age_groups = np.random.choice(['18-25', '26-40', '41-60', '60+'], size=n_samples, p=[0.20, 0.40, 0.25, 0.15])

    # Peer median amounts per claim type
    peer_medians_map = {'auto': 3500.0, 'health': 5000.0, 'property': 8500.0}
    peer_medians = np.array([peer_medians_map[ct] * np.random.uniform(0.8, 1.2) for ct in claim_types])

    # Base claim amounts with normal variation
    amount_multipliers = np.random.lognormal(mean=0.0, sigma=0.25, size=n_samples)
    claim_amounts = np.round(peer_medians * amount_multipliers, 2)

    # Identifiers
    phones = [f"555-{random.randint(100, 999):03d}-{random.randint(1000, 9999):04d}" for _ in range(n_samples)]
    addresses = [f"{random.randint(100, 9999)} {random.choice(['Main St', 'Oak Ave', 'Maple Dr', 'Pine Rd', 'Elm St'])}, City" for _ in range(n_samples)]
    bank_accounts = [f"ACCT-{random.randint(10000000, 99999999)}" for _ in range(n_samples)]
    garages = [f"FAC-{random.randint(1000, 1200):04d}" for _ in range(n_samples)]

    # Structured flags
    num_prior_claims = np.random.poisson(lam=0.6, size=n_samples)
    injury_flags = np.random.binomial(n=1, p=0.20, size=n_samples)
    police_flags = np.random.binomial(n=1, p=0.30, size=n_samples)

    # Base narratives
    narrative_templates = [
        "Vehicle sustained front bumper damage during low speed collision at intersection.",
        "Patient visited outpatient facility for routine medical evaluation and diagnostic testing.",
        "Water leak caused damage to hardwood flooring in primary living room area.",
        "Minor rear-end collision on highway during heavy traffic. No injuries reported at scene.",
        "Claimant reported storm damage to roof tiles following severe rainfall event."
    ]
    narratives = [random.choice(narrative_templates) for _ in range(n_samples)]

    # Build DataFrame
    df = pd.DataFrame({
        'claim_id': claim_ids,
        'policy_id': policy_ids,
        'claimant_id': claimant_ids,
        'provider_id': provider_ids,
        'claim_date': [d.strftime('%Y-%m-%d') for d in claim_dates],
        'policy_start_date': [d.strftime('%Y-%m-%d') for d in policy_start_dates],
        'claim_type': claim_types,
        'claim_amount': claim_amounts,
        'peer_median_amount': np.round(peer_medians, 2),
        'phone': phones,
        'address': addresses,
        'bank_account': bank_accounts,
        'garage_or_hospital_id': garages,
        'narrative': narratives,
        'num_prior_claims': num_prior_claims,
        'injury_flag': injury_flags,
        'police_report_flag': police_flags,
        'region': regions,
        'age_group': age_groups,
        # Hidden tag labels (0 or 1)
        'tag_DUPLICATE_CLAIM': 0,
        'tag_INFLATED_AMOUNT': 0,
        'tag_SUSPICIOUS_TIMING': 0,
        'tag_PROVIDER_ANOMALY': 0,
        'tag_NARRATIVE_INCONSISTENCY': 0,
        'tag_RING_SUSPECT': 0,
        'is_novel_pattern': 0
    })

    # Convert date strings to datetime for manipulation
    df['c_date_dt'] = pd.to_datetime(df['claim_date'])
    df['p_date_dt'] = pd.to_datetime(df['policy_start_date'])

    # --- Fraud Injection ---
    n_inflated = max(1, int(0.009 * n_samples))
    n_timing = max(1, int(0.009 * n_samples))
    n_narrative = max(1, int(0.009 * n_samples))
    n_duplicate = max(1, int(0.007 * n_samples))

    # 1. INFLATED_AMOUNT
    inflated_idx = np.random.choice(df.index, size=n_inflated, replace=False)
    for idx in inflated_idx:
        mult = np.random.uniform(2.5, 4.8)
        df.at[idx, 'claim_amount'] = np.round(df.at[idx, 'peer_median_amount'] * mult, 2)
        df.at[idx, 'tag_INFLATED_AMOUNT'] = 1

    # 2. SUSPICIOUS_TIMING
    timing_idx = np.random.choice(df.index, size=n_timing, replace=False)
    for idx in timing_idx:
        lag_days = random.randint(1, 25)
        df.at[idx, 'p_date_dt'] = df.at[idx, 'c_date_dt'] - timedelta(days=lag_days)
        df.at[idx, 'policy_start_date'] = df.at[idx, 'p_date_dt'].strftime('%Y-%m-%d')
        df.at[idx, 'tag_SUSPICIOUS_TIMING'] = 1

    # 3. PROVIDER_ANOMALY (~15 specific malicious providers)
    anomaly_providers = [f"PRV-{i:04d}" for i in range(1485, 1500)]
    provider_mask = df['provider_id'].isin(anomaly_providers)
    prov_indices = df[provider_mask].index
    for idx in prov_indices:
        df.at[idx, 'claim_amount'] = np.round(df.at[idx, 'claim_amount'] * np.random.uniform(2.0, 3.5), 2)
        df.at[idx, 'tag_PROVIDER_ANOMALY'] = 1

    # 4. NARRATIVE_INCONSISTENCY
    narrative_idx = np.random.choice(df.index, size=n_narrative, replace=False)
    inconsistent_texts = [
        ("No injuries sustained, occupants walked away without medical need.", 1, 0), # injury_flag = 1 contradicts text
        ("Police arrived immediately and filed an official collision report at scene.", 0, 0), # police_flag = 0 contradicts text
        ("Minor scratching on side mirror with zero structural vehicle damage.", 0, 1) # amount inconsistency set below
    ]
    for idx in narrative_idx:
        text, set_inj, set_pol = random.choice(inconsistent_texts)
        df.at[idx, 'narrative'] = text
        df.at[idx, 'injury_flag'] = set_inj
        df.at[idx, 'police_report_flag'] = set_pol
        if set_inj == 0 and set_pol == 1:
            df.at[idx, 'claim_amount'] = np.round(df.at[idx, 'peer_median_amount'] * 4.2, 2)
        df.at[idx, 'tag_NARRATIVE_INCONSISTENCY'] = 1

    # 5. DUPLICATE_CLAIM
    dup_sources = np.random.choice(df.index, size=n_duplicate, replace=False)
    for i, src_idx in enumerate(dup_sources):
        # Create a duplicate claim near in time
        target_idx = (src_idx + 1) % len(df)
        df.at[target_idx, 'claimant_id'] = df.at[src_idx, 'claimant_id']
        df.at[target_idx, 'claim_amount'] = df.at[src_idx, 'claim_amount']
        df.at[target_idx, 'narrative'] = df.at[src_idx, 'narrative']
        df.at[target_idx, 'c_date_dt'] = df.at[src_idx, 'c_date_dt'] + timedelta(days=random.randint(1, 10))
        df.at[target_idx, 'claim_date'] = df.at[target_idx, 'c_date_dt'].strftime('%Y-%m-%d')
        df.at[target_idx, 'tag_DUPLICATE_CLAIM'] = 1

    # 6. RING_SUSPECT (~30 rings of 5 to 15 claimants)
    # 25 standard rings + 5 novel pattern rings in test period
    n_standard_rings = min(25, max(1, n_samples // 2000))
    n_novel_rings = min(5, max(1, n_samples // 10000))
    
    # Standard rings across whole timeline
    for r in range(n_standard_rings):
        ring_size = min(len(df), random.randint(5, 12))
        ring_phone = f"555-999-{r:04d}"
        ring_bank = f"ACCT-RING-{r:04d}"
        ring_address = f"RING-ADDR-{r:04d} Boulevard"
        ring_indices = np.random.choice(df.index, size=ring_size, replace=False)
        for idx in ring_indices:
            df.at[idx, 'phone'] = ring_phone
            df.at[idx, 'bank_account'] = ring_bank
            df.at[idx, 'address'] = ring_address
            df.at[idx, 'tag_RING_SUSPECT'] = 1

    # Novel rings restricted to late 2024 (Test period)
    test_mask = df['c_date_dt'] >= datetime(2024, 7, 1)
    test_indices = df[test_mask].index
    if len(test_indices) > 0:
        for r in range(n_novel_rings):
            ring_size = min(len(test_indices), random.randint(5, 15))
            ring_phone = f"555-NOVEL-{r:04d}"
            ring_bank = f"ACCT-NOVEL-{r:04d}"
            ring_garage = f"FAC-NOVEL-{r:04d}"
            chosen_indices = np.random.choice(test_indices, size=ring_size, replace=False)
            for idx in chosen_indices:
                df.at[idx, 'phone'] = ring_phone
                df.at[idx, 'bank_account'] = ring_bank
                df.at[idx, 'garage_or_hospital_id'] = ring_garage
                df.at[idx, 'tag_RING_SUSPECT'] = 1
                df.at[idx, 'is_novel_pattern'] = 1

    # Noise injection (make clean claims slightly noisy)
    clean_mask = (df[['tag_DUPLICATE_CLAIM', 'tag_INFLATED_AMOUNT', 'tag_SUSPICIOUS_TIMING', 
                      'tag_PROVIDER_ANOMALY', 'tag_NARRATIVE_INCONSISTENCY', 'tag_RING_SUSPECT']].sum(axis=1) == 0)
    clean_indices = df[clean_mask].index
    n_noise = min(len(clean_indices), max(1, int(0.03 * n_samples)))
    noise_indices = np.random.choice(clean_indices, size=n_noise, replace=False)
    for idx in noise_indices:
        # Add 1.7x amount or 33-day timing to legitimate claims to test false positive rates
        r = random.random()
        if r < 0.33:
            df.at[idx, 'claim_amount'] = np.round(df.at[idx, 'peer_median_amount'] * 1.75, 2)
        elif r < 0.66:
            df.at[idx, 'p_date_dt'] = df.at[idx, 'c_date_dt'] - timedelta(days=32)
            df.at[idx, 'policy_start_date'] = df.at[idx, 'p_date_dt'].strftime('%Y-%m-%d')

    # Calculate overall fraud label
    tag_cols = ['tag_DUPLICATE_CLAIM', 'tag_INFLATED_AMOUNT', 'tag_SUSPICIOUS_TIMING', 
                'tag_PROVIDER_ANOMALY', 'tag_NARRATIVE_INCONSISTENCY', 'tag_RING_SUSPECT']
    df['is_fraud'] = (df[tag_cols].sum(axis=1) > 0).astype(int)

    # Clean up temporary helper columns
    df.drop(columns=['c_date_dt', 'p_date_dt'], inplace=True)
    return df

def get_time_splits(df: pd.DataFrame):
    """
    Time-based split:
    - Train: first 24 months (2022-01-01 to 2023-12-31)
    - Validation: next 6 months (2024-01-01 to 2024-06-30)
    - Test: last 6 months (2024-07-01 to 2024-12-31)
    - Novel Pattern Test: test subset where is_novel_pattern == 1
    """
    df_sorted = df.copy()
    df_sorted['dt'] = pd.to_datetime(df_sorted['claim_date'])
    
    train_mask = (df_sorted['dt'] >= '2022-01-01') & (df_sorted['dt'] <= '2023-12-31')
    val_mask = (df_sorted['dt'] >= '2024-01-01') & (df_sorted['dt'] <= '2024-06-30')
    test_mask = (df_sorted['dt'] >= '2024-07-01') & (df_sorted['dt'] <= '2024-12-31')

    df_train = df_sorted[train_mask].drop(columns=['dt'])
    df_val = df_sorted[val_mask].drop(columns=['dt'])
    df_test = df_sorted[test_mask].drop(columns=['dt'])
    df_novel = df_test[df_test['is_novel_pattern'] == 1].copy()

    return df_train, df_val, df_test, df_novel

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    print("Generating 50,000 synthetic claims...")
    df_all = generate_synthetic_claims(50000, seed=42)
    df_train, df_val, df_test, df_novel = get_time_splits(df_all)
    
    print(f"Total dataset shape: {df_all.shape}")
    print(f"Train set (2022-2023): {len(df_train)} rows, Fraud rate: {df_train['is_fraud'].mean():.4f}")
    print(f"Val set (2024 H1): {len(df_val)} rows, Fraud rate: {df_val['is_fraud'].mean():.4f}")
    print(f"Test set (2024 H2): {len(df_test)} rows, Fraud rate: {df_test['is_fraud'].mean():.4f}")
    print(f"Novel pattern test set: {len(df_novel)} rows, Fraud rate: {df_novel['is_fraud'].mean():.4f}")

    df_all.to_parquet('data/claims_all.parquet', index=False)
    print("Saved raw dataset to data/claims_all.parquet.")
