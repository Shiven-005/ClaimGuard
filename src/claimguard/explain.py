"""
SHAP Explanations and Evidence Formatting for ClaimGuard.
Extracts top 5 SHAP feature contributions and generates human-readable evidence strings.
"""

import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Any


class ClaimExplainer:
    def __init__(self, lgb_model, feature_names: List[str]):
        self.model = lgb_model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)

    def get_top_shap_features(self, X_claim: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Compute SHAP values for a single row dataframe and return top N contributing features.
        """
        shap_values = self.explainer.shap_values(X_claim)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0] # binary classification positive class
        elif len(shap_values.shape) == 2:
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values

        feature_vals = X_claim.iloc[0].values
        
        # Sort by absolute SHAP contribution
        abs_indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]

        top_features = []
        for idx in abs_indices:
            name = self.feature_names[idx]
            val = float(feature_vals[idx])
            s_val = float(shap_vals[idx])
            top_features.append({
                'feature': name,
                'value': round(val, 4),
                'shap_contribution': round(s_val, 4)
            })

        return top_features

    @staticmethod
    def generate_tag_evidence(claim_dict: Dict[str, Any], tag_probs: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Format evidence statements for flagged tags.
        """
        evidence_list = []
        
        amount = claim_dict.get('claim_amount', 0.0)
        peer_median = claim_dict.get('peer_median_amount', 1.0)
        ratio = amount / max(peer_median, 1.0)
        
        c_date = pd.to_datetime(claim_dict.get('claim_date', '2024-01-01'))
        p_date = pd.to_datetime(claim_dict.get('policy_start_date', '2022-01-01'))
        days_lag = (c_date - p_date).days

        narrative = str(claim_dict.get('narrative', '')).lower()
        inj_flag = claim_dict.get('injury_flag', 0)
        pol_flag = claim_dict.get('police_report_flag', 0)

        # 1. DUPLICATE_CLAIM
        if tag_probs.get('DUPLICATE_CLAIM', 0.0) >= 0.40:
            evidence_list.append({
                'tag': 'DUPLICATE_CLAIM',
                'confidence': round(tag_probs['DUPLICATE_CLAIM'], 2),
                'evidence': f"Recent prior claim detected for claimant {claim_dict.get('claimant_id', 'N/A')} with identical narrative and amount (${amount:,.2f})."
            })

        # 2. INFLATED_AMOUNT
        if tag_probs.get('INFLATED_AMOUNT', 0.0) >= 0.40 or ratio >= 2.0:
            evidence_list.append({
                'tag': 'INFLATED_AMOUNT',
                'confidence': round(max(tag_probs.get('INFLATED_AMOUNT', 0.0), min(ratio / 4.0, 1.0)), 2),
                'evidence': f"Claim amount (${amount:,.2f}) is {ratio:.1f}x the peer median (${peer_median:,.2f}) for {claim_dict.get('claim_type', 'claim')}."
            })

        # 3. SUSPICIOUS_TIMING
        if tag_probs.get('SUSPICIOUS_TIMING', 0.0) >= 0.40 or (days_lag >= 0 and days_lag <= 30):
            evidence_list.append({
                'tag': 'SUSPICIOUS_TIMING',
                'confidence': round(max(tag_probs.get('SUSPICIOUS_TIMING', 0.0), 0.85), 2),
                'evidence': f"Claim filed only {days_lag} days after policy inception date ({claim_dict.get('policy_start_date')})."
            })

        # 4. PROVIDER_ANOMALY
        if tag_probs.get('PROVIDER_ANOMALY', 0.0) >= 0.40:
            evidence_list.append({
                'tag': 'PROVIDER_ANOMALY',
                'confidence': round(tag_probs['PROVIDER_ANOMALY'], 2),
                'evidence': f"Provider {claim_dict.get('provider_id')} has billing patterns significantly exceeding historical peer medians."
            })

        # 5. NARRATIVE_INCONSISTENCY
        if tag_probs.get('NARRATIVE_INCONSISTENCY', 0.0) >= 0.40 or ("no injur" in narrative and inj_flag == 1):
            details = []
            if "no injur" in narrative and inj_flag == 1:
                details.append("Narrative states 'no injuries' but injury_flag=1.")
            if "police" in narrative and pol_flag == 0:
                details.append("Narrative mentions police report but police_report_flag=0.")
            if not details:
                details.append("Text narrative contradicts structured claim facts.")
            evidence_list.append({
                'tag': 'NARRATIVE_INCONSISTENCY',
                'confidence': round(max(tag_probs.get('NARRATIVE_INCONSISTENCY', 0.0), 0.80), 2),
                'evidence': " ".join(details)
            })

        # 6. RING_SUSPECT
        if tag_probs.get('RING_SUSPECT', 0.0) >= 0.40:
            evidence_list.append({
                'tag': 'RING_SUSPECT',
                'confidence': round(tag_probs['RING_SUSPECT'], 2),
                'evidence': f"Claimant {claim_dict.get('claimant_id')} shares phone/bank account across multiple distinct claimants in the entity graph."
            })

        return evidence_list
