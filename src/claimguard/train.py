"""
Training and Evaluation Pipeline for ClaimGuard.
Trains LightGBM classifier, probability calibrator, multi-label tag classifier,
tunes decision policy thresholds, and evaluates metrics and ablations.
Generates artifacts/metrics.json and plots.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, recall_score, precision_score, f1_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import joblib

from claimguard.synth import generate_synthetic_claims, get_time_splits
from claimguard.features import FeaturePipeline
from claimguard.weak_labels import generate_weak_label_matrix, TAG_NAMES

REVIEW_COST = 150.0 # currency units per review

def calculate_net_savings(y_true: np.ndarray, y_prob: np.ndarray, amounts: np.ndarray, threshold: float) -> float:
    """
    Net Savings = Sum(Fraud Amount Caught) - (Review Cost * Number of Reviews)
    """
    flagged = (y_prob >= threshold)
    fraud_caught_amount = np.sum(amounts[flagged & (y_true == 1)])
    num_reviews = np.sum(flagged)
    total_cost = num_reviews * REVIEW_COST
    return float(fraud_caught_amount - total_cost)

def optimize_thresholds(y_val: np.ndarray, p_val: np.ndarray, val_amounts: np.ndarray) -> dict:
    """
    Tune review and escalation thresholds on validation set to maximize net savings.
    """
    best_savings = -float('inf')
    best_review_thresh = 0.35
    best_escalate_thresh = 0.70

    threshold_grid = np.linspace(0.10, 0.90, 81)
    savings_list = []

    for t in threshold_grid:
        sav = calculate_net_savings(y_val, p_val, val_amounts, t)
        savings_list.append(sav)
        if sav > best_savings:
            best_savings = sav
            best_review_thresh = float(t)

    # Escalation threshold is set for top high-risk tier (higher precision)
    best_escalate_thresh = min(0.95, max(best_review_thresh + 0.25, 0.75))

    return {
        'review_threshold': best_review_thresh,
        'escalate_threshold': best_escalate_thresh,
        'val_max_net_savings': best_savings
    }

def run_training_and_eval():
    os.makedirs('artifacts/plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    print("Step 1: Generating/Loading Datasets...")
    data_path = 'data/claims_all.parquet'
    if os.path.exists(data_path):
        df_all = pd.read_parquet(data_path)
    else:
        df_all = generate_synthetic_claims(50000, seed=42)
        df_all.to_parquet(data_path, index=False)

    df_train, df_val, df_test, df_novel = get_time_splits(df_all)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}, Novel Test: {len(df_novel)}")

    print("\nStep 2: Fitting Feature Pipeline...")
    feat_pipeline = FeaturePipeline()
    X_train = feat_pipeline.fit_transform(df_train)
    X_val = feat_pipeline.transform(df_val)
    X_test = feat_pipeline.transform(df_test)
    X_novel = feat_pipeline.transform(df_novel)

    y_train = df_train['is_fraud'].values
    y_val = df_val['is_fraud'].values
    y_test = df_test['is_fraud'].values
    y_novel = df_novel['is_fraud'].values

    print("\nStep 3: Training Main LightGBM Fraud Model...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_train, lgb_val]
    )

    p_val_raw = model.predict(X_val)
    p_test_raw = model.predict(X_test)
    p_novel_raw = model.predict(X_novel)

    print("Step 4: Calibrating Probabilities (Isotonic Regression)...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(p_val_raw, y_val)

    p_val = calibrator.transform(p_val_raw)
    p_test = calibrator.transform(p_test_raw)
    p_novel = calibrator.transform(p_novel_raw)

    print("\nStep 5: Training Multi-label Tag Classifier...")
    # Generate weak labels for training set
    wl_train = generate_weak_label_matrix(df_train)
    Y_tags_train = wl_train[TAG_NAMES].values

    tag_base_clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    tag_classifier = MultiOutputClassifier(tag_base_clf)
    tag_classifier.fit(X_train, Y_tags_train)

    # Predict tags on test set
    tag_probs_test = np.array([m.predict_proba(X_test)[:, 1] for m in tag_classifier.estimators_]).T
    Y_tags_test_gt = df_test[[f"tag_{t}" for t in TAG_NAMES]].values

    print("\nStep 6: Optimizing Decision Policy Thresholds...")
    threshold_policy = optimize_thresholds(y_val, p_val, df_val['claim_amount'].values)
    review_thresh = threshold_policy['review_threshold']
    print(f"Optimal Review Threshold: {review_thresh:.4f}, Net Val Savings: ${threshold_policy['val_max_net_savings']:,.2f}")

    print("\nStep 7: Evaluating Test Metrics & Ablation Studies...")
    # Main model metrics on Test set
    roc_auc = float(roc_auc_score(y_test, p_test))
    precisions, recalls, _ = precision_recall_curve(y_test, p_test)
    pr_auc = float(auc(recalls, precisions))

    # Recall at 90% precision
    rec_at_90p = 0.0
    for p, r in zip(precisions, recalls):
        if p >= 0.90:
            if r > rec_at_90p:
                rec_at_90p = float(r)

    # Precision@Top-5%
    top_5_percent_k = int(0.05 * len(p_test))
    top_5_idx = np.argsort(p_test)[::-1][:top_5_percent_k]
    prec_at_top5 = float(np.mean(y_test[top_5_idx]))

    # Money Saved on Test Set
    test_savings = calculate_net_savings(y_test, p_test, df_test['claim_amount'].values, review_thresh)

    # Per-tag metrics
    tag_metrics = {}
    tag_f1s = []
    for i, tag in enumerate(TAG_NAMES):
        tag_pred = (tag_probs_test[:, i] >= 0.40).astype(int)
        tag_gt = Y_tags_test_gt[:, i]
        p_t = float(precision_score(tag_gt, tag_pred, zero_division=0))
        r_t = float(recall_score(tag_gt, tag_pred, zero_division=0))
        f1_t = float(f1_score(tag_gt, tag_pred, zero_division=0))
        tag_f1s.append(f1_t)
        tag_metrics[tag] = {
            'precision': round(p_t, 4),
            'recall': round(r_t, 4),
            'f1': round(f1_t, 4)
        }

    macro_f1 = float(np.mean(tag_f1s))
    micro_f1 = float(f1_score(Y_tags_test_gt.ravel(), (tag_probs_test.ravel() >= 0.40).astype(int), zero_division=0))

    # Baseline Model: Logistic Regression on raw tabular numerical fields
    print("Evaluating Baseline Logistic Regression...")
    tab_cols = [c for c in X_train.columns if not (c.startswith('graph_') or c.startswith('narrative_') or c.startswith('rule_') or c == 'anomaly_score')]
    lr_baseline = LogisticRegression(max_iter=300)
    lr_baseline.fit(X_train[tab_cols], y_train)
    p_test_lr = lr_baseline.predict_proba(X_test[tab_cols])[:, 1]
    lr_roc = float(roc_auc_score(y_test, p_test_lr))
    lr_pr, lr_rec, _ = precision_recall_curve(y_test, p_test_lr)
    lr_pr_auc = float(auc(lr_rec, lr_pr))

    # Ablations
    print("Running Feature Ablation Study...")
    ablation_results = {}
    
    # 1. Tabular only
    cols_tab = [c for c in X_train.columns if not (c.startswith('graph_') or c.startswith('narrative_') or c.startswith('rule_') or c == 'anomaly_score')]
    m_tab = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1).fit(X_train[cols_tab], y_train)
    p_tab = m_tab.predict_proba(X_test[cols_tab])[:, 1]
    pr_tab, r_tab, _ = precision_recall_curve(y_test, p_tab)
    ablation_results['Tabular Only'] = {'roc_auc': round(float(roc_auc_score(y_test, p_tab)), 4), 'pr_auc': round(float(auc(r_tab, pr_tab)), 4)}

    # 2. Tabular + Text
    cols_text = cols_tab + [c for c in X_train.columns if c.startswith('narrative_') or c.startswith('rule_')]
    m_text = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1).fit(X_train[cols_text], y_train)
    p_text = m_text.predict_proba(X_test[cols_text])[:, 1]
    pr_txt, r_txt, _ = precision_recall_curve(y_test, p_text)
    ablation_results['Tabular + Text'] = {'roc_auc': round(float(roc_auc_score(y_test, p_text)), 4), 'pr_auc': round(float(auc(r_txt, pr_txt)), 4)}

    # 3. Tabular + Graph
    cols_graph = cols_tab + [c for c in X_train.columns if c.startswith('graph_')]
    m_graph = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1).fit(X_train[cols_graph], y_train)
    p_graph = m_graph.predict_proba(X_test[cols_graph])[:, 1]
    pr_grp, r_grp, _ = precision_recall_curve(y_test, p_graph)
    ablation_results['Tabular + Graph'] = {'roc_auc': round(float(roc_auc_score(y_test, p_graph)), 4), 'pr_auc': round(float(auc(r_grp, pr_grp)), 4)}

    # 4. Tabular + Anomaly
    cols_anom = cols_tab + ['anomaly_score']
    m_anom = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=42, verbose=-1).fit(X_train[cols_anom], y_train)
    p_anom = m_anom.predict_proba(X_test[cols_anom])[:, 1]
    pr_anm, r_anm, _ = precision_recall_curve(y_test, p_anom)
    ablation_results['Tabular + Anomaly'] = {'roc_auc': round(float(roc_auc_score(y_test, p_anom)), 4), 'pr_auc': round(float(auc(r_anm, pr_anm)), 4)}

    # 5. Full Model
    ablation_results['Full Model'] = {'roc_auc': round(roc_auc, 4), 'pr_auc': round(pr_auc, 4)}

    # Novel Pattern Test Evaluation
    novel_roc = float(roc_auc_score(y_novel, p_novel)) if len(np.unique(y_novel)) > 1 else 1.0
    novel_rec = float(recall_score(y_novel, (p_novel >= review_thresh).astype(int)))
    novel_prec = float(precision_score(y_novel, (p_novel >= review_thresh).astype(int), zero_division=1))

    # Fairness Check (Flag rate across region & age_group)
    test_flagged = (p_test >= review_thresh).astype(int)
    fairness_df = df_test.assign(flagged=test_flagged)
    fairness_region = fairness_df.groupby('region')['flagged'].mean().to_dict()
    fairness_age = fairness_df.groupby('age_group')['flagged'].mean().to_dict()

    # Robustness Test (Simulate fraudsters splitting claim amounts by 50%)
    df_test_split = df_test.copy()
    df_test_split['claim_amount'] = df_test_split['claim_amount'] * 0.50
    X_test_split = feat_pipeline.transform(df_test_split)
    p_test_split = calibrator.transform(model.predict(X_test_split))
    orig_rec = float(recall_score(y_test, (p_test >= review_thresh).astype(int)))
    split_rec = float(recall_score(y_test, (p_test_split >= review_thresh).astype(int)))
    detection_drop = round(orig_rec - split_rec, 4)

    # Combine all metrics into JSON structure
    metrics_summary = {
        'overall_test_metrics': {
            'pr_auc': round(pr_auc, 4),
            'roc_auc': round(roc_auc, 4),
            'recall_at_90_precision': round(rec_at_90p, 4),
            'precision_at_top_5_percent': round(prec_at_top5, 4),
            'estimated_money_saved': round(test_savings, 2),
            'review_threshold': round(review_thresh, 4),
            'escalate_threshold': round(threshold_policy['escalate_threshold'], 4)
        },
        'tag_metrics': {
            'per_tag': tag_metrics,
            'macro_f1': round(macro_f1, 4),
            'micro_f1': round(micro_f1, 4)
        },
        'baseline_logistic_regression': {
            'roc_auc': round(lr_roc, 4),
            'pr_auc': round(lr_pr_auc, 4)
        },
        'ablation_study': ablation_results,
        'novel_pattern_test': {
            'sample_count': int(len(y_novel)),
            'roc_auc': round(novel_roc, 4),
            'recall': round(novel_rec, 4),
            'precision': round(novel_prec, 4)
        },
        'fairness_check': {
            'flag_rate_by_region': {k: round(v, 4) for k, v in fairness_region.items()},
            'flag_rate_by_age_group': {k: round(v, 4) for k, v in fairness_age.items()}
        },
        'robustness_test': {
            'original_recall': round(orig_rec, 4),
            'amount_split_50pct_recall': round(split_rec, 4),
            'recall_drop': detection_drop
        }
    }

    # Save metrics.json
    with open('artifacts/metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print("\nSuccessfully written artifacts/metrics.json!")

    print("Step 8: Generating Plots...")
    # 1. Calibration Curve
    plt.figure(figsize=(6, 5))
    prob_true, prob_pred = calibration_curve(y_test, p_test, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='Calibrated LightGBM')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.title('Probability Calibration Curve')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.tight_layout()
    plt.savefig('artifacts/plots/calibration_curve.png', dpi=200)
    plt.close()

    # 2. Precision-Recall Curve
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, label=f'LightGBM (PR-AUC = {pr_auc:.3f})')
    plt.plot(lr_rec, lr_pr, linestyle='--', label=f'Baseline LR (PR-AUC = {lr_pr_auc:.3f})')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    plt.savefig('artifacts/plots/pr_curve.png', dpi=200)
    plt.close()

    print("Step 9: Saving Serialized Model Artifacts...")
    joblib.dump(feat_pipeline, 'artifacts/feature_pipeline.joblib')
    joblib.dump(model, 'artifacts/lgb_model.joblib')
    joblib.dump(calibrator, 'artifacts/calibrator.joblib')
    joblib.dump(tag_classifier, 'artifacts/tag_classifier.joblib')
    joblib.dump(threshold_policy, 'artifacts/policy_thresholds.joblib')

    print("All artifacts successfully trained and saved to artifacts/")

if __name__ == '__main__':
    run_training_and_eval()
