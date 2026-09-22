"""
Tests for Feature Pipeline and Graph feature computation.
"""

import pandas as pd
from claimguard.synth import generate_synthetic_claims
from claimguard.features import FeaturePipeline
from claimguard.graph import extract_graph_features

def test_graph_features():
    df = generate_synthetic_claims(100, seed=42)
    graph_feats = extract_graph_features(df)
    assert isinstance(graph_feats, pd.DataFrame)
    assert len(graph_feats) == 100
    assert 'graph_component_size' in graph_feats.columns
    assert 'graph_max_shared_count' in graph_feats.columns

def test_feature_pipeline_fit_transform():
    df = generate_synthetic_claims(100, seed=42)
    pipeline = FeaturePipeline()
    X = pipeline.fit_transform(df)
    assert isinstance(X, pd.DataFrame)
    assert len(X) == 100
    assert 'amount_to_peer_ratio' in X.columns
    assert 'narrative_inconsistency_score' in X.columns
    assert 'anomaly_score' in X.columns
