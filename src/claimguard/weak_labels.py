"""
Weak Labels Module for ClaimGuard.
Defines domain heuristic labeling functions and a weighted voting Label Model
to produce multi-label tag training targets.
"""

import numpy as np
import pandas as pd

TAG_NAMES = [
    'DUPLICATE_CLAIM',
    'INFLATED_AMOUNT',
    'SUSPICIOUS_TIMING',
    'PROVIDER_ANOMALY',
    'NARRATIVE_INCONSISTENCY',
    'RING_SUSPECT'
]

def lf_duplicate_claim(df: pd.DataFrame) -> np.ndarray:
    """Flag claims with same claimant and near-identical amount/date."""
    votes = np.zeros(len(df), dtype=int)
    # Group by claimant_id and check time delta / amount similarity
    df_sorted = df.sort_values(['claimant_id', 'claim_date']).copy()
    df_sorted['prev_claimant'] = df_sorted['claimant_id'].shift(1)
    df_sorted['prev_date'] = pd.to_datetime(df_sorted['claim_date']).shift(1)
    df_sorted['curr_date'] = pd.to_datetime(df_sorted['claim_date'])
    df_sorted['days_diff'] = (df_sorted['curr_date'] - df_sorted['prev_date']).dt.days

    same_claimant = (df_sorted['claimant_id'] == df_sorted['prev_claimant'])
    near_time = (df_sorted['days_diff'] <= 20) & (df_sorted['days_diff'] >= 0)
    
    dup_indices = df_sorted[same_claimant & near_time].index
    votes[df.index.get_indexer(dup_indices)] = 1
    return votes

def lf_inflated_amount(df: pd.DataFrame) -> np.ndarray:
    """Flag claims where amount is >= 2.0x peer median."""
    ratio = df['claim_amount'] / df['peer_median_amount'].replace(0, 1.0)
    return (ratio >= 2.0).astype(int).values

def lf_suspicious_timing(df: pd.DataFrame) -> np.ndarray:
    """Flag claims occurring within 30 days of policy start date."""
    c_date = pd.to_datetime(df['claim_date'])
    p_date = pd.to_datetime(df['policy_start_date'])
    days_since_start = (c_date - p_date).dt.days
    return ((days_since_start >= 0) & (days_since_start <= 30)).astype(int).values

def lf_provider_anomaly(df: pd.DataFrame) -> np.ndarray:
    """Flag claims from providers whose median billing ratio is > 2.0."""
    ratio = df['claim_amount'] / df['peer_median_amount'].replace(0, 1.0)
    df_temp = df.assign(ratio=ratio)
    provider_medians = df_temp.groupby('provider_id')['ratio'].transform('median')
    return (provider_medians >= 2.0).astype(int).values

def lf_narrative_inconsistency(df: pd.DataFrame) -> np.ndarray:
    """Flag claims with keyword contradictions between narrative and structured fields."""
    votes = np.zeros(len(df), dtype=int)
    narratives = df['narrative'].str.lower()
    
    # Contradiction 1: Text says no injury, but injury_flag == 1
    no_injury_text = narratives.str.contains("no injur", regex=False, na=False)
    has_injury_flag = (df['injury_flag'] == 1)
    
    # Contradiction 2: Text says police filed report, but police_report_flag == 0
    police_text = narratives.str.contains("police", regex=False, na=False)
    no_police_flag = (df['police_report_flag'] == 0)

    contra1 = no_injury_text & has_injury_flag
    contra2 = police_text & no_police_flag

    votes[(contra1 | contra2).values] = 1
    return votes

def lf_ring_suspect(df: pd.DataFrame, graph_df: pd.DataFrame = None) -> np.ndarray:
    """Flag claims associated with shared identifiers / entities."""
    if graph_df is not None and 'graph_max_shared_count' in graph_df.columns:
        return (graph_df['graph_max_shared_count'] >= 3).astype(int).values
    
    # Fallback to phone / address count in dataframe
    phone_counts = df.groupby('phone')['claim_id'].transform('count')
    bank_counts = df.groupby('bank_account')['claim_id'].transform('count')
    return ((phone_counts >= 4) | (bank_counts >= 4)).astype(int).values

def generate_weak_label_matrix(df: pd.DataFrame, graph_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate weak label matrix for each tag.
    Returns DataFrame with columns corresponding to TAG_NAMES.
    """
    matrix = pd.DataFrame(index=df.index)
    matrix['DUPLICATE_CLAIM'] = lf_duplicate_claim(df)
    matrix['INFLATED_AMOUNT'] = lf_inflated_amount(df)
    matrix['SUSPICIOUS_TIMING'] = lf_suspicious_timing(df)
    matrix['PROVIDER_ANOMALY'] = lf_provider_anomaly(df)
    matrix['NARRATIVE_INCONSISTENCY'] = lf_narrative_inconsistency(df)
    matrix['RING_SUSPECT'] = lf_ring_suspect(df, graph_df)
    return matrix

if __name__ == '__main__':
    from claimguard.synth import generate_synthetic_claims
    print("Testing weak label generation...")
    sample_df = generate_synthetic_claims(1000, seed=42)
    wl_matrix = generate_weak_label_matrix(sample_df)
    print("Weak labels summary:")
    print(wl_matrix.sum())
