"""
Inference Module for ClaimGuard.
Loads trained model artifacts and provides single claim & batch prediction services.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from claimguard.explain import ClaimExplainer
from claimguard.weak_labels import TAG_NAMES

class ClaimPredictor:
    def __init__(self, artifact_dir: str = 'artifacts'):
        self.artifact_dir = artifact_dir
        self.pipeline = joblib.load(os.path.join(artifact_dir, 'feature_pipeline.joblib'))
        self.model = joblib.load(os.path.join(artifact_dir, 'lgb_model.joblib'))
        self.calibrator = joblib.load(os.path.join(artifact_dir, 'calibrator.joblib'))
        self.tag_classifier = joblib.load(os.path.join(artifact_dir, 'tag_classifier.joblib'))
        self.thresholds = joblib.load(os.path.join(artifact_dir, 'policy_thresholds.joblib'))
        
        self.explainer = ClaimExplainer(self.model, self.pipeline.feature_names)
        self.review_thresh = self.thresholds.get('review_threshold', 0.35)
        self.escalate_thresh = self.thresholds.get('escalate_threshold', 0.70)

    def _determine_action(self, prob: float) -> str:
        if prob >= self.escalate_thresh:
            return "ESCALATE"
        elif prob >= self.review_thresh:
            return "REVIEW"
        else:
            return "AUTO_APPROVE"

    def predict_single(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single claim dictionary.
        Returns risk score (0-100), action, tags with evidence, and top SHAP features.
        """
        df_single = pd.DataFrame([claim])
        X_single = self.pipeline.transform(df_single)

        # Main risk score
        p_raw = self.model.predict(X_single)[0]
        prob_calibrated = float(self.calibrator.transform([p_raw])[0])
        risk_score = round(prob_calibrated * 100.0, 1)

        action = self._determine_action(prob_calibrated)

        # Multi-label tag predictions
        tag_probs_raw = np.array([m.predict_proba(X_single)[:, 1][0] for m in self.tag_classifier.estimators_])
        tag_probs_dict = {tag: float(prob) for tag, prob in zip(TAG_NAMES, tag_probs_raw)}

        # Evidence generation
        tags_with_evidence = self.explainer.generate_tag_evidence(claim, tag_probs_dict)

        # Top SHAP features
        top_shap = self.explainer.get_top_shap_features(X_single, top_n=5)

        return {
            'claim_id': claim.get('claim_id', 'CLM-UNKNOWN'),
            'risk_score': risk_score,
            'raw_probability': round(prob_calibrated, 4),
            'action': action,
            'tags': tags_with_evidence,
            'top_shap_features': top_shap
        }

    def predict_batch(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a batch of claims and return DataFrame with risk scores, actions, and tags.
        """
        X_batch = self.pipeline.transform(claims_df)
        p_raw = self.model.predict(X_batch)
        probs_calibrated = self.calibrator.transform(p_raw)

        actions = [self._determine_action(p) for p in probs_calibrated]
        risk_scores = np.round(probs_calibrated * 100.0, 1)

        # Multi-label tag predictions
        tag_probs_mat = np.array([m.predict_proba(X_batch)[:, 1] for m in self.tag_classifier.estimators_]).T

        flagged_tags_list = []
        top_evidence_list = []

        for idx, row in claims_df.reset_index(drop=True).iterrows():
            tag_probs_dict = {tag: float(tag_probs_mat[idx, i]) for i, tag in enumerate(TAG_NAMES)}
            evidences = self.explainer.generate_tag_evidence(row.to_dict(), tag_probs_dict)
            
            tags_str = ", ".join([e['tag'] for e in evidences]) if evidences else "NONE"
            evidence_str = "; ".join([e['evidence'] for e in evidences]) if evidences else "No flags triggered."

            flagged_tags_list.append(tags_str)
            top_evidence_list.append(evidence_str)

        res_df = claims_df.copy()
        res_df['risk_score'] = risk_scores
        res_df['action'] = actions
        res_df['flagged_tags'] = flagged_tags_list
        res_df['evidence'] = top_evidence_list
        return res_df

_predictor_instance = None

def get_predictor(artifact_dir: str = 'artifacts') -> ClaimPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ClaimPredictor(artifact_dir)
    return _predictor_instance

def score_single(claim: Dict[str, Any]) -> Dict[str, Any]:
    predictor = get_predictor()
    return predictor.predict_single(claim)

def score_batch(claims_df: pd.DataFrame) -> pd.DataFrame:
    predictor = get_predictor()
    return predictor.predict_batch(claims_df)

if __name__ == '__main__':
    print("Testing Predictor...")
    test_claim = {
        'claim_id': 'CLM-TEST-001',
        'policy_id': 'POL-99999',
        'claimant_id': 'CLMNT-99999',
        'provider_id': 'PRV-1490',
        'claim_date': '2024-08-01',
        'policy_start_date': '2024-07-20',
        'claim_type': 'auto',
        'claim_amount': 14500.0,
        'peer_median_amount': 3500.0,
        'phone': '555-999-0001',
        'address': '123 Test St',
        'bank_account': 'ACCT-TEST-001',
        'garage_or_hospital_id': 'FAC-1001',
        'narrative': 'No injuries sustained during collision.',
        'num_prior_claims': 2,
        'injury_flag': 1,
        'police_report_flag': 0,
        'region': 'North',
        'age_group': '26-40'
    }

    res = score_single(test_claim)
    print("Single Claim Score Result:")
    import json
    print(json.dumps(res, indent=2))
