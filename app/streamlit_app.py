"""
Streamlit Dashboard for ClaimGuard.
Interactive UI featuring single claim scoring, batch CSV processing, investigator queue,
model performance analytics, and system architecture.
"""

import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from claimguard.predict import get_predictor, score_single, score_batch
from claimguard.synth import generate_synthetic_claims

st.set_page_config(
    page_title="ClaimGuard - Insurance Claim Flagging System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .badge-auto {
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
    }
    .badge-review {
        background-color: #FEF3C7;
        color: #B45309;
        font-weight: 700;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
    }
    .badge-escalate {
        background-color: #FEE2E2;
        color: #B91C1C;
        font-weight: 700;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://img.icons8.com/isometric/96/shield.png", width=64)
st.sidebar.title("🛡️ ClaimGuard")
st.sidebar.caption("AI-Powered Claim Flagging & Tagging")

page = st.sidebar.radio(
    "Navigation",
    ["1. Score Single Claim", "2. Batch Upload (CSV)", "3. Investigator Queue", "4. Model Report", "5. About & Architecture"]
)

# Load artifacts / predictor
predictor = get_predictor()

# --- PRESETS FOR PAGE 1 ---
PRESET_CLEAN = {
    'claim_id': 'CLM-DEMO-001',
    'policy_id': 'POL-10042',
    'claimant_id': 'CLMNT-10099',
    'provider_id': 'PRV-1005',
    'claim_date': '2024-08-01',
    'policy_start_date': '2022-03-15',
    'claim_type': 'auto',
    'claim_amount': 3200.0,
    'peer_median_amount': 3500.0,
    'phone': '555-123-4567',
    'address': '742 Evergreen Terrace',
    'bank_account': 'ACCT-990112',
    'garage_or_hospital_id': 'FAC-1002',
    'narrative': 'Minor bumper scratch while parking in grocery store parking lot. No injuries reported.',
    'num_prior_claims': 0,
    'injury_flag': 0,
    'police_report_flag': 0,
    'region': 'North',
    'age_group': '26-40'
}

PRESET_INFLATED_TIMING = {
    'claim_id': 'CLM-DEMO-002',
    'policy_id': 'POL-88123',
    'claimant_id': 'CLMNT-44012',
    'provider_id': 'PRV-1490',
    'claim_date': '2024-08-01',
    'policy_start_date': '2024-07-20',
    'claim_type': 'auto',
    'claim_amount': 15800.0,
    'peer_median_amount': 3500.0,
    'phone': '555-987-6543',
    'address': '101 Pine Road',
    'bank_account': 'ACCT-441099',
    'garage_or_hospital_id': 'FAC-1199',
    'narrative': 'Vehicle sustained severe front collision on highway late night. Total loss claimed.',
    'num_prior_claims': 3,
    'injury_flag': 0,
    'police_report_flag': 0,
    'region': 'South',
    'age_group': '18-25'
}

PRESET_RING = {
    'claim_id': 'CLM-DEMO-003',
    'policy_id': 'POL-99401',
    'claimant_id': 'CLMNT-77102',
    'provider_id': 'PRV-1488',
    'claim_date': '2024-08-10',
    'policy_start_date': '2023-01-10',
    'claim_type': 'property',
    'claim_amount': 28500.0,
    'peer_median_amount': 8500.0,
    'phone': '555-999-0004', # Ring phone
    'address': 'RING-ADDR-0004 Boulevard',
    'bank_account': 'ACCT-RING-0004',
    'garage_or_hospital_id': 'FAC-1050',
    'narrative': 'Water pipe burst causing damage across multiple rooms.',
    'num_prior_claims': 4,
    'injury_flag': 0,
    'police_report_flag': 1,
    'region': 'East',
    'age_group': '41-60'
}

# ---------------------------------------------------------
# PAGE 1: SCORE SINGLE CLAIM
# ---------------------------------------------------------
if page == "1. Score Single Claim":
    st.markdown('<div class="main-header">🛡️ Single Claim Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate an individual claim for fraud risk, multi-label tags, and SHAP evidence.</div>', unsafe_allow_html=True)

    st.subheader("Load Preset Sample")
    col_p1, col_p2, col_p3 = st.columns(3)
    
    if 'claim_data' not in st.session_state:
        st.session_state['claim_data'] = PRESET_CLEAN

    if col_p1.button("🟢 Preset: Clean Claim"):
        st.session_state['claim_data'] = PRESET_CLEAN
    if col_p2.button("🔴 Preset: Inflated + Timing"):
        st.session_state['claim_data'] = PRESET_INFLATED_TIMING
    if col_p3.button("⚠️ Preset: Fraud Ring Suspect"):
        st.session_state['claim_data'] = PRESET_RING

    curr = st.session_state['claim_data']

    with st.form("single_claim_form"):
        st.subheader("Claim Details")
        c1, c2, c3 = st.columns(3)
        claim_id = c1.text_input("Claim ID", value=curr['claim_id'])
        policy_id = c2.text_input("Policy ID", value=curr['policy_id'])
        claimant_id = c3.text_input("Claimant ID", value=curr['claimant_id'])

        c4, c5, c6 = st.columns(3)
        claim_type = c4.selectbox("Claim Type", ['auto', 'health', 'property'], index=['auto', 'health', 'property'].index(curr['claim_type']))
        claim_amount = c5.number_input("Claim Amount ($)", value=float(curr['claim_amount']), step=100.0)
        peer_median_amount = c6.number_input("Peer Median Amount ($)", value=float(curr['peer_median_amount']), step=100.0)

        c7, c8 = st.columns(2)
        claim_date = c7.date_input("Claim Filing Date", pd.to_datetime(curr['claim_date'])).strftime('%Y-%m-%d')
        policy_start_date = c8.date_input("Policy Inception Date", pd.to_datetime(curr['policy_start_date'])).strftime('%Y-%m-%d')

        c9, c10, c11 = st.columns(3)
        provider_id = c9.text_input("Provider ID", value=curr['provider_id'])
        phone = c10.text_input("Phone Number", value=curr['phone'])
        bank_account = c11.text_input("Bank Account", value=curr['bank_account'])

        c12, c13, c14 = st.columns(3)
        address = c12.text_input("Address", value=curr['address'])
        garage_id = c13.text_input("Garage / Hospital ID", value=curr['garage_or_hospital_id'])
        num_prior_claims = c14.number_input("Num Prior Claims", value=int(curr['num_prior_claims']), min_value=0)

        c15, c16, c17 = st.columns(3)
        injury_flag = c15.selectbox("Injury Reported Flag", [0, 1], index=int(curr['injury_flag']))
        police_flag = c16.selectbox("Police Report Filed Flag", [0, 1], index=int(curr['police_report_flag']))
        region = c17.selectbox("Region", ['North', 'South', 'East', 'West', 'Central'], index=['North', 'South', 'East', 'West', 'Central'].index(curr['region']))

        narrative = st.text_area("Claim Narrative Text", value=curr['narrative'], height=80)

        submit_btn = st.form_submit_button("🔍 Evaluate Claim Risk", use_container_width=True)

    if submit_btn:
        input_dict = {
            'claim_id': claim_id,
            'policy_id': policy_id,
            'claimant_id': claimant_id,
            'provider_id': provider_id,
            'claim_date': str(claim_date),
            'policy_start_date': str(policy_start_date),
            'claim_type': claim_type,
            'claim_amount': float(claim_amount),
            'peer_median_amount': float(peer_median_amount),
            'phone': phone,
            'address': address,
            'bank_account': bank_account,
            'garage_or_hospital_id': garage_id,
            'narrative': narrative,
            'num_prior_claims': int(num_prior_claims),
            'injury_flag': int(injury_flag),
            'police_report_flag': int(police_flag),
            'region': region,
            'age_group': '26-40'
        }

        with st.spinner("Scoring claim and computing SHAP explanations..."):
            res = score_single(input_dict)

        st.markdown("---")
        st.subheader("🎯 Evaluation Results")

        res_col1, res_col2, res_col3 = st.columns(3)

        res_col1.metric("Risk Score", f"{res['risk_score']} / 100")
        
        act = res['action']
        if act == "AUTO_APPROVE":
            res_col2.markdown('### Action:<br><span class="badge-auto">🟢 AUTO_APPROVE</span>', unsafe_allow_html=True)
        elif act == "REVIEW":
            res_col2.markdown('### Action:<br><span class="badge-review">⚠️ REVIEW</span>', unsafe_allow_html=True)
        else:
            res_col2.markdown('### Action:<br><span class="badge-escalate">🚨 ESCALATE</span>', unsafe_allow_html=True)

        res_col3.metric("Raw Probability", f"{res['raw_probability']:.4f}")

        st.markdown("### 🏷️ Multi-Label Tag Evidence")
        if res['tags']:
            for t in res['tags']:
                st.warning(f"**[{t['tag']}]** (Confidence: {t['confidence']*100:.0f}%)\n\n👉 {t['evidence']}")
        else:
            st.success("✅ No suspicious tag flags triggered for this claim.")

        st.markdown("### 📊 Top 5 SHAP Feature Contributions")
        shap_df = pd.DataFrame(res['top_shap_features'])
        
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = ['#EF4444' if x > 0 else '#3B82F6' for x in shap_df['shap_contribution']]
        ax.barh(shap_df['feature'], shap_df['shap_contribution'], color=colors)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel("SHAP Impact on Fraud Risk")
        ax.set_title("Top 5 Driving Risk Factors")
        plt.tight_layout()
        st.pyplot(fig)

# ---------------------------------------------------------
# PAGE 2: BATCH UPLOAD (CSV)
# ---------------------------------------------------------
elif page == "2. Batch Upload (CSV)":
    st.markdown('<div class="main-header">📂 Batch CSV Claim Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a batch CSV file to score thousands of claims instantly.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Claims CSV File", type=["csv"])
    
    st.caption("Or click below to load a sample batch of 1,000 synthetic test claims:")
    if st.button("Generate & Load Sample Batch (1,000 Claims)"):
        with st.spinner("Generating sample claims..."):
            sample_df = generate_synthetic_claims(1000, seed=123)
            st.session_state['batch_df'] = sample_df

    if uploaded_file is not None:
        st.session_state['batch_df'] = pd.read_csv(uploaded_file)

    if 'batch_df' in st.session_state:
        batch_input_df = st.session_state['batch_df']
        st.write(f"Loaded **{len(batch_input_df)}** claims for scoring:")
        st.dataframe(batch_input_df.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Scoring Pipeline", type="primary", use_container_width=True):
            with st.spinner(f"Scoring {len(batch_input_df)} claims across tabular, text, and graph models..."):
                scored_df = score_batch(batch_input_df)

            st.session_state['scored_results'] = scored_df

    if 'scored_results' in st.session_state:
        res_df = st.session_state['scored_results']
        st.markdown("---")
        st.subheader("Batch Results Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Claims", len(res_df))
        
        flagged_n = int((res_df['action'] != 'AUTO_APPROVE').sum())
        m2.metric("Flagged for Review/Escalation", f"{flagged_n} ({flagged_n/len(res_df)*100:.1f}%)")
        
        total_exp = res_df.loc[res_df['action'] != 'AUTO_APPROVE', 'claim_amount'].sum()
        m3.metric("Flagged Exposure ($)", f"${total_exp:,.2f}")
        
        m4.metric("Average Risk Score", f"{res_df['risk_score'].mean():.1f} / 100")

        st.subheader("Flagged Results Table")
        st.dataframe(res_df[['claim_id', 'risk_score', 'action', 'claim_amount', 'flagged_tags', 'evidence']].head(100), use_container_width=True)

        csv_data = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Flagged Results CSV",
            data=csv_data,
            file_name="claimguard_flagged_results.csv",
            mime="text/csv",
            use_container_width=True
        )

# ---------------------------------------------------------
# PAGE 3: INVESTIGATOR QUEUE
# ---------------------------------------------------------
elif page == "3. Investigator Queue":
    st.markdown('<div class="main-header">📋 Investigator Priority Queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Claims ranked by Expected Loss = (Risk Score / 100) × Claim Amount</div>', unsafe_allow_html=True)

    if 'queue_df' not in st.session_state:
        with st.spinner("Loading investigator queue from test dataset..."):
            sample_df = generate_synthetic_claims(2000, seed=42)
            scored = score_batch(sample_df)
            scored['expected_loss'] = (scored['risk_score'] / 100.0) * scored['claim_amount']
            st.session_state['queue_df'] = scored.sort_values('expected_loss', ascending=False)

    queue = st.session_state['queue_df']

    c_f1, c_f2 = st.columns(2)
    action_filter = c_f1.multiselect("Filter by Action", ['ESCALATE', 'REVIEW', 'AUTO_APPROVE'], default=['ESCALATE', 'REVIEW'])
    
    filtered_queue = queue[queue['action'].isin(action_filter)]

    st.write(f"Showing **{len(filtered_queue)}** flagged claims in priority order:")
    
    display_cols = ['claim_id', 'action', 'risk_score', 'claim_amount', 'expected_loss', 'flagged_tags', 'evidence']
    st.dataframe(filtered_queue[display_cols].head(50), use_container_width=True)

    st.markdown("### Inspect High-Risk Claim")
    claim_choice = st.selectbox("Select Claim ID to Inspect", filtered_queue['claim_id'].head(20).tolist())
    
    if claim_choice:
        row = filtered_queue[filtered_queue['claim_id'] == claim_choice].iloc[0].to_dict()
        with st.expander(f"🔍 Case Details for {claim_choice}", expanded=True):
            e1, e2, e3 = st.columns(3)
            e1.markdown(f"**Risk Score**: `{row['risk_score']} / 100`")
            e2.markdown(f"**Action**: `{row['action']}`")
            e3.markdown(f"**Expected Loss**: `${row['expected_loss']:,.2f}`")
            
            st.markdown(f"**Narrative**: *\"{row['narrative']}\"*")
            st.markdown(f"**Triggered Flags**: `{row['flagged_tags']}`")
            st.markdown(f"**Evidence Breakdown**: {row['evidence']}")

# ---------------------------------------------------------
# PAGE 4: MODEL REPORT
# ---------------------------------------------------------
elif page == "4. Model Report":
    st.markdown('<div class="main-header">📈 Model Performance & Ablation Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation metrics, feature ablations, calibration, and fairness audits on test data.</div>', unsafe_allow_html=True)

    metrics_file = 'artifacts/metrics.json'
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        st.subheader("Overall Test Performance")
        m1, m2, m3, m4, m5 = st.columns(5)
        ov = metrics['overall_test_metrics']
        m1.metric("PR-AUC", ov['pr_auc'])
        m2.metric("ROC-AUC", ov['roc_auc'])
        m3.metric("Recall @ 90% Prec.", ov['recall_at_90_precision'])
        m4.metric("Precision @ Top 5%", ov['precision_at_top_5_percent'])
        m5.metric("Est. Money Saved", f"${ov['estimated_money_saved']:,.2f}")

        st.markdown("---")
        c_p1, c_p2 = st.columns(2)

        with c_p1:
            st.subheader("Calibration & PR Curves")
            if os.path.exists('artifacts/plots/pr_curve.png'):
                st.image('artifacts/plots/pr_curve.png', caption="Precision-Recall Curve vs Baseline")
            if os.path.exists('artifacts/plots/calibration_curve.png'):
                st.image('artifacts/plots/calibration_curve.png', caption="Probability Calibration Curve")

        with c_p2:
            st.subheader("Feature Ablation Study")
            ablation_df = pd.DataFrame(metrics['ablation_study']).T
            st.table(ablation_df)

            st.subheader("Per-Tag Precision, Recall & F1")
            tag_df = pd.DataFrame(metrics['tag_metrics']['per_tag']).T
            st.table(tag_df)

        st.markdown("---")
        st.subheader("Fairness & Robustness Audits")
        f1, f2 = st.columns(2)
        with f1:
            st.write("**Flag Rate Across Regions**")
            st.json(metrics['fairness_check']['flag_rate_by_region'])
        with f2:
            st.write("**Robustness (50% Amount Split Attack)**")
            rob = metrics['robustness_test']
            st.write(f"- Original Recall: `{rob['original_recall']}`")
            st.write(f"- Post-Split Recall: `{rob['amount_split_50pct_recall']}`")
            st.write(f"- Recall Drop: `{rob['recall_drop']}`")

    else:
        st.warning("`artifacts/metrics.json` not found. Please run `python -m claimguard.train` first.")

# ---------------------------------------------------------
# PAGE 5: ABOUT & ARCHITECTURE
# ---------------------------------------------------------
elif page == "5. About & Architecture":
    st.markdown('<div class="main-header">ℹ️ About ClaimGuard</div>', unsafe_allow_html=True)
    st.markdown("""
    **ClaimGuard** is an open-source, production-ready Insurance Claim Flagging and Tagging System designed for CPU hosting.
    
    ### System Architecture
    ```
    +-------------------------+     +------------------------+     +-----------------------+
    |  Synthetic Claim Data   | --> |  Feature Pipeline      | --> |  LightGBM Classifier  |
    |  (50k claims, 3 years)  |     |  (Tabular, NLP, Graph) |     |  + Calibrator        |
    +-------------------------+     +------------------------+     +-----------------------+
                                                                               |
                                                                               v
    +-------------------------+     +------------------------+     +-----------------------+
    | FastAPI / Streamlit UI  | <-- | SHAP & Evidence Engine | <-- | Decision Policy       |
    | (Port 7860 HF Space)    |     | (Multi-label Tags)     |     | (Net Savings Tuning)  |
    +-------------------------+     +------------------------+     +-----------------------+
    ```

    ### Multi-Label Tag Taxonomy
    1. **DUPLICATE_CLAIM**: Resubmitted claims near in date or with identical amounts/narratives.
    2. **INFLATED_AMOUNT**: Billed amount significantly exceeds peer median for claim type and procedure.
    3. **SUSPICIOUS_TIMING**: Claim filed within 30 days of policy inception.
    4. **PROVIDER_ANOMALY**: Provider billing patterns consistently exceeding historical baseline.
    5. **NARRATIVE_INCONSISTENCY**: Discrepancy between text narrative and structured flags (e.g., text injury claim vs flag).
    6. **RING_SUSPECT**: Shared entity graph (phones, addresses, bank accounts) across distinct claimants.

    ### How to Swap in Real Data
    1. Replace `synth.py` with your database connector (SQL, Snowflake, S3).
    2. Map your columns to the `ClaimInputSchema` format.
    3. Run `python -m claimguard.train` to re-train LightGBM, Graph, and NLP models on your data.
    """)
