"""
Feature Engineering Module for ClaimGuard.
Builds tabular, NLP (TF-IDF + rules), graph, and anomaly detection features.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

from claimguard.graph import extract_graph_features

class FeaturePipeline:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        self.nlp_clf = LogisticRegression(C=1.0, max_iter=200)
        self.iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.provider_stats = {}
        self.is_fitted = False

    def _compute_tabular_features(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        df_out = pd.DataFrame(index=df.index)

        # Amounts
        df_out['amount_to_peer_ratio'] = df['claim_amount'] / df['peer_median_amount'].replace(0, 1.0)
        df_out['amount_diff'] = df['claim_amount'] - df['peer_median_amount']
        df_out['claim_amount'] = df['claim_amount']
        df_out['peer_median_amount'] = df['peer_median_amount']

        # Dates
        c_date = pd.to_datetime(df['claim_date'])
        p_date = pd.to_datetime(df['policy_start_date'])
        days_lag = (c_date - p_date).dt.days
        df_out['days_since_policy_start'] = np.clip(days_lag, 0, 1095)
        df_out['claim_weekday'] = c_date.dt.weekday
        df_out['claim_month'] = c_date.dt.month
        df_out['is_weekend'] = (c_date.dt.weekday >= 5).astype(int)

        # Historical frequencies
        if is_train:
            self.claimant_freq_30d = {}
            self.claimant_freq_90d = {}
            self.claimant_freq_365d = {}
            # Compute frequency per claimant
            counts_365 = df.groupby('claimant_id')['claim_id'].count().to_dict()
            self.claimant_freq_365d = counts_365

        df_out['claims_per_claimant_365d'] = df['claimant_id'].map(self.claimant_freq_365d).fillna(1)
        df_out['claims_per_claimant_30d'] = np.minimum(df_out['claims_per_claimant_365d'], 2)
        df_out['claims_per_claimant_90d'] = np.minimum(df_out['claims_per_claimant_365d'], 3)

        # Provider z-score vs peers
        if is_train:
            p_means = df.groupby('provider_id')['claim_amount'].mean()
            p_stds = df.groupby('provider_id')['claim_amount'].std().fillna(1.0)
            global_mean = df['claim_amount'].mean()
            global_std = df['claim_amount'].std()
            self.provider_stats = {'means': p_means.to_dict(), 'stds': p_stds.to_dict(), 'g_mean': global_mean, 'g_std': global_std}

        g_mean = self.provider_stats.get('g_mean', 4000.0)
        g_std = self.provider_stats.get('g_std', 2000.0)
        p_means_dict = self.provider_stats.get('means', {})
        p_stds_dict = self.provider_stats.get('stds', {})

        p_mean_col = df['provider_id'].map(p_means_dict).fillna(g_mean)
        p_std_col = df['provider_id'].map(p_stds_dict).fillna(g_std).replace(0, g_std)
        df_out['provider_zscore'] = (df['claim_amount'] - p_mean_col) / p_std_col

        # Flags and priors
        df_out['num_prior_claims'] = df['num_prior_claims']
        df_out['injury_flag'] = df['injury_flag']
        df_out['police_report_flag'] = df['police_report_flag']

        # Categoricals (One-Hot)
        cat_cols = ['claim_type', 'region', 'age_group']
        if is_train:
            cat_feats = self.ohe.fit_transform(df[cat_cols])
        else:
            cat_feats = self.ohe.transform(df[cat_cols])
        
        cat_df = pd.DataFrame(cat_feats, columns=self.ohe.get_feature_names_out(cat_cols), index=df.index)
        df_out = pd.concat([df_out, cat_df], axis=1)

        return df_out

    def _compute_text_features(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        df_out = pd.DataFrame(index=df.index)
        narratives = df['narrative'].fillna("")

        if is_train:
            tfidf_mat = self.tfidf.fit_transform(narratives)
            # Train NLP classifier target based on synthetic inconsistency rule
            no_inj_text = narratives.str.lower().str.contains("no injur", regex=False)
            target = (no_inj_text & (df['injury_flag'] == 1)).astype(int)
            self.nlp_clf.fit(tfidf_mat, target)
        else:
            tfidf_mat = self.tfidf.transform(narratives)

        df_out['narrative_inconsistency_score'] = self.nlp_clf.predict_proba(tfidf_mat)[:, 1]

        # Rule features
        narr_lower = narratives.str.lower()
        df_out['rule_narrative_no_injury_flag_mismatch'] = (narr_lower.str.contains("no injur", regex=False) & (df['injury_flag'] == 1)).astype(int)
        df_out['rule_narrative_police_flag_mismatch'] = (narr_lower.str.contains("police", regex=False) & (df['police_report_flag'] == 0)).astype(int)
        df_out['narrative_len'] = narratives.str.len()

        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit feature pipeline on training data and transform."""
        print("Fitting Tabular Features...")
        tab_df = self._compute_tabular_features(df, is_train=True)
        
        print("Fitting NLP Features...")
        text_df = self._compute_text_features(df, is_train=True)

        print("Fitting Graph Features...")
        graph_df = extract_graph_features(df)

        print("Fitting IsolationForest Anomaly Model...")
        num_cols = tab_df.select_dtypes(include=[np.number]).columns
        self.iso_forest.fit(tab_df[num_cols])
        iso_df = pd.DataFrame({'anomaly_score': self.iso_forest.score_samples(tab_df[num_cols])}, index=df.index)

        X_full = pd.concat([tab_df, text_df, graph_df, iso_df], axis=1)
        self.feature_names = X_full.columns.tolist()
        self.is_fitted = True
        return X_full

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform validation or test data."""
        if not self.is_fitted:
            raise ValueError("FeaturePipeline must be fitted before transform.")

        tab_df = self._compute_tabular_features(df, is_train=False)
        text_df = self._compute_text_features(df, is_train=False)
        graph_df = extract_graph_features(df)

        num_cols = [c for c in tab_df.select_dtypes(include=[np.number]).columns if c in tab_df.columns]
        iso_df = pd.DataFrame({'anomaly_score': self.iso_forest.score_samples(tab_df[num_cols])}, index=df.index)

        X_full = pd.concat([tab_df, text_df, graph_df, iso_df], axis=1)
        
        # Ensure exact column alignment
        for col in self.feature_names:
            if col not in X_full.columns:
                X_full[col] = 0.0
        X_full = X_full[self.feature_names]
        return X_full

if __name__ == '__main__':
    from claimguard.synth import generate_synthetic_claims
    print("Testing FeaturePipeline...")
    sample_df = generate_synthetic_claims(500, seed=42)
    pipeline = FeaturePipeline()
    X = pipeline.fit_transform(sample_df)
    print(f"Extracted feature matrix shape: {X.shape}")
    print(f"Top 10 feature names: {list(X.columns[:10])}")
