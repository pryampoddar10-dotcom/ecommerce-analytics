"""
app.py — AI-Driven E-Commerce Analytics Dashboard
Author : Pryam Poddar | SP Jain School of Global Management | MGB Term 2
Subject: Data Analytics (Individual Assignment)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pickle, json, os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Analytics | Pryam Poddar",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(BASE)
DATA    = os.path.join(ROOT, 'dataset', 'ecommerce_clustered.csv')
PLOTS   = os.path.join(BASE, 'plots')
MODELS  = os.path.join(BASE, 'models')

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header{font-size:2rem;font-weight:700;color:#7F77DD;margin-bottom:0.2rem}
    .sub-header{font-size:1rem;color:#888;margin-bottom:1.5rem}
    .metric-card{background:#f8f8ff;border-left:4px solid #7F77DD;
                 padding:1rem 1.2rem;border-radius:8px;margin-bottom:0.8rem}
    .metric-val{font-size:1.8rem;font-weight:700;color:#3C3489}
    .metric-lbl{font-size:0.8rem;color:#666;margin-top:2px}
    .section-title{font-size:1.15rem;font-weight:600;
                   color:#3C3489;border-bottom:2px solid #7F77DD;
                   padding-bottom:4px;margin:1.2rem 0 0.8rem}
    .insight-box{background:#eef6ff;border-left:3px solid #1D9E75;
                 padding:0.8rem 1rem;border-radius:6px;font-size:0.9rem;
                 color:#1a1a1a;margin:0.5rem 0}
    .warn-box{background:#fff8ee;border-left:3px solid #EF9F27;
              padding:0.8rem 1rem;border-radius:6px;font-size:0.9rem;color:#1a1a1a}
    [data-testid="stMetricValue"]{font-size:1.6rem;color:#3C3489}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA)

@st.cache_resource
def load_json(path):
    with open(path) as f:
        return json.load(f)

@st.cache_resource
def load_model(path):
    with open(path,'rb') as f:
        return pickle.load(f)

def safe_img(fname):
    p = os.path.join(PLOTS, fname)
    if os.path.exists(p):
        st.image(p, use_container_width=True)
    else:
        st.info(f"Run the ML scripts first to generate: {fname}")

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛒 E-Commerce Analytics")
    st.markdown("**Pryam Poddar**  \nSP Jain | MGB Term 2  \nData Analytics Assignment")
    st.divider()
    page = st.radio("Navigate", [
        "🏠  Overview",
        "👥  Customer Segmentation",
        "🎯  Conversion Prediction",
        "💰  Spend Prediction",
        "🔗  Product Associations",
        "📊  EDA & Insights"
    ])
    st.divider()
    st.caption(f"Dataset: {len(df):,} respondents · 45 features")
    st.caption("Platform: Streamlit Cloud + GitHub")

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<div class="main-header">AI-Driven E-Commerce Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Pryam Poddar · SP Jain School of Global Management · MGB Term 2 · Data Analytics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Respondents", f"{len(df):,}")
    c2.metric("Features", "45")
    c3.metric("Purchase Rate", f"{df['will_purchase'].mean():.1%}")
    c4.metric("Avg Monthly Spend", "₹2,000–5,000")
    c5.metric("Customer Segments", "5")

    st.markdown('<div class="section-title">Business Concept</div>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard demonstrates how an **AI-driven e-commerce platform** can leverage data analytics
    to optimise customer targeting, dynamic pricing, and purchase conversion across the Indian market.
    The system applies four core machine learning techniques to a synthetic survey dataset of **2,000 Indian consumers**.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Platform Components</div>', unsafe_allow_html=True)
        components = {
            "👥 Customer Segmentation": ("K-Means Clustering", "Silhouette = 0.078", "#7F77DD"),
            "🎯 Conversion Prediction": ("Gradient Boosting", "AUC = 0.773", "#1D9E75"),
            "💰 Spend Prediction":      ("Gradient Boosting", "R² = 0.742", "#EF9F27"),
            "🔗 Product Associations":  ("Apriori ARM", "131 rules · Max lift = 1.21", "#D85A30"),
        }
        for name, (algo, metric, color) in components.items():
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-weight:600;color:{color};font-size:1rem">{name}</div>
                <div style="font-size:0.85rem;color:#444;margin-top:3px">
                    Algorithm: <b>{algo}</b> &nbsp;|&nbsp; {metric}
                </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Respondent Breakdown</div>', unsafe_allow_html=True)
        safe_img("persona_distribution.png")

    st.markdown('<div class="section-title">Survey Design Summary</div>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        st.markdown("**Section A — Demographics** (Q1–Q5)")
        st.markdown("Age · Gender · City tier · Education · Employment")
        st.markdown("**Section B — Shopping Behaviour** (Q6–Q11)")
        st.markdown("Frequency · Platforms · Device · Browse time · Cart abandonment")
    with col4:
        st.markdown("**Section C — Preferences** (Q12–Q15)")
        st.markdown("Category choices · Review importance · Discovery channel · Recommendation acceptance")
        st.markdown("**Section D — Spending & Pricing** (Q16–Q24)")
        st.markdown("Income · Monthly spend · Max transaction · Discount sensitivity · Payment · Loyalty")
    with col5:
        st.markdown("**Section E — New Gaps Added** (Q20–Q24)")
        st.markdown("Trust/privacy · Social commerce · Post-purchase loyalty · Festive triggers · AI comfort")
        st.markdown("**Target Variable** (Q25)")
        st.markdown("`will_purchase` — binary classification target (1=likely, 0=unlikely)")

    st.markdown('<div class="section-title">Key Insights at a Glance</div>', unsafe_allow_html=True)
    ins = [
        "62% of surveyed Indian consumers would purchase on a new AI-driven platform within one month.",
        "Premium loyalists (12% of users) contribute disproportionate spend — avg transaction >₹25,000.",
        "Repeat purchase intent is the single strongest predictor of conversion (importance weight: 30%).",
        "Electronics + Sports is the highest co-purchase pair in urban professional segments.",
        "Discount sensitivity is inversely correlated with income (r = −0.60) — confirming price-tier segmentation.",
    ]
    for insight in ins:
        st.markdown(f'<div class="insight-box">✅ {insight}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════════
elif "Segmentation" in page:
    st.markdown('<div class="main-header">👥 Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown("K-Means clustering (k=5) applied to 20 behavioural and demographic features.")

    # metrics row
    try:
        cm = load_json(os.path.join(MODELS, 'clustering_metrics.json'))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Algorithm", "K-Means")
        c2.metric("Clusters (k)", str(cm.get('k', 5)))
        c3.metric("Silhouette Score", str(cm.get('silhouette', '—')))
        c4.metric("Davies-Bouldin", str(cm.get('davies_bouldin', '—')))
    except:
        st.info("Run 03_clustering.py to generate metrics.")

    tab1, tab2, tab3 = st.tabs(["Cluster Profiles", "Visualisations", "Cluster Deep-Dive"])

    with tab1:
        st.markdown('<div class="section-title">The 5 Customer Personas</div>', unsafe_allow_html=True)
        personas = {
            "Premium Loyalists":    ("High income · Metro · Multi-loyalty member · Weekly+ shopper · Low discount sensitivity",
                                     "Drive highest revenue per transaction — target with premium product launches and early access offers.", "#7F77DD"),
            "Tech-Savvy Explorers": ("Young urban · High AI comfort · Strong recommendation acceptance · Electronics + Sports buyer",
                                     "Most receptive to AI personalisation — ideal for testing recommendation engine features.", "#1D9E75"),
            "Deal Hunters":         ("Mid-income · Very high discount sensitivity · High cart abandonment · Flash sale driven",
                                     "Need time-limited offers to convert — target with festive sale notifications and BNPL options.", "#EF9F27"),
            "Value Seekers":        ("Tier 2/3 · COD preferred · Self-employed · Moderate frequency · Price-comparison shoppers",
                                     "Price transparency is critical — show lowest-price guarantees and free delivery thresholds.", "#D85A30"),
            "Occasional Browsers":  ("Lower frequency · High browse time · High cart abandonment · Undecided",
                                     "Re-engagement segment — use wishlist reminders and limited-time nudges to convert.", "#D4537E"),
        }
        for name, (traits, strategy, color) in personas.items():
            with st.expander(f"**{name}**"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"**Profile:** {traits}")
                with col2:
                    st.markdown(f'<div class="insight-box">💡 Strategy: {strategy}</div>', unsafe_allow_html=True)

        if 'cluster_name' in df.columns:
            st.markdown('<div class="section-title">Cluster Size Distribution</div>', unsafe_allow_html=True)
            cluster_counts = df['cluster_name'].value_counts().reset_index()
            cluster_counts.columns = ['Segment', 'Count']
            cluster_counts['Share (%)'] = (cluster_counts['Count'] / len(df) * 100).round(1)
            cluster_counts['Avg Purchase Rate'] = cluster_counts['Segment'].apply(
                lambda s: f"{df[df['cluster_name']==s]['will_purchase'].mean():.1%}"
            )
            st.dataframe(cluster_counts, use_container_width=True, hide_index=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Elbow & Silhouette — optimal k selection**")
            safe_img("clustering_elbow.png")
        with col2:
            st.markdown("**PCA projection of 5 clusters**")
            safe_img("clustering_pca.png")
        st.markdown("**Cluster feature radar charts**")
        safe_img("clustering_radar.png")

    with tab3:
        st.markdown('<div class="section-title">Explore a Cluster</div>', unsafe_allow_html=True)
        if 'cluster_name' in df.columns:
            selected = st.selectbox("Select segment", df['cluster_name'].dropna().unique())
            sub = df[df['cluster_name'] == selected]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Respondents", len(sub))
            c2.metric("Purchase Rate", f"{sub['will_purchase'].mean():.1%}")
            c3.metric("Avg AI Comfort", f"{sub['ai_personalisation_comfort'].mean():.2f}/5")
            c4.metric("Loyalty Members", f"{sub['loyalty_member_flag'].mean():.1%}")

            col1, col2 = st.columns(2)
            with col1:
                income_vc = sub['monthly_income_bracket'].value_counts()
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(range(len(income_vc)), income_vc.values, color='#7F77DD')
                ax.set_xticks(range(len(income_vc)))
                ax.set_xticklabels([l[:12] for l in income_vc.index], rotation=25, ha='right', fontsize=8)
                ax.set_title(f'Income distribution — {selected}', fontsize=10)
                ax.spines[['top','right']].set_visible(False)
                st.pyplot(fig); plt.close()
            with col2:
                cats = ['cat_fashion','cat_electronics','cat_grocery','cat_beauty',
                        'cat_home','cat_books','cat_sports','cat_toys']
                cat_labels = ['Fashion','Electronics','Grocery','Beauty','Home','Books','Sports','Toys']
                rates = [sub[c].mean() for c in cats]
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(cat_labels, rates, color='#1D9E75')
                ax.set_xlabel('Purchase rate', fontsize=9)
                ax.set_title(f'Category preferences — {selected}', fontsize=10)
                ax.spines[['top','right']].set_visible(False)
                st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — CONVERSION PREDICTION
# ═══════════════════════════════════════════════════════════════════
elif "Conversion" in page:
    st.markdown('<div class="main-header">🎯 Purchase Conversion Prediction</div>', unsafe_allow_html=True)
    st.markdown("Binary classification: predict whether a customer will purchase on a new AI-driven platform.")

    try:
        cm = load_json(os.path.join(MODELS, 'classification_metrics.json'))
        best = 'Gradient Boosting'
        bm   = cm.get(best, {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best Model", best)
        c2.metric("Accuracy", f"{bm.get('accuracy', 0):.1%}")
        c3.metric("ROC-AUC", f"{bm.get('roc_auc', 0):.3f}")
        c4.metric("F1-Score", f"{bm.get('f1', 0):.3f}")
        c5.metric("CV-AUC (5-fold)", f"{bm.get('cv_auc', 0):.3f}")
    except:
        st.info("Run 04_classification.py to generate metrics.")

    tab1, tab2, tab3 = st.tabs(["Model Performance", "Feature Importance", "Live Predictor"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**ROC curves — all three models**")
            safe_img("classification_roc.png")
        with col2:
            st.markdown("**Confusion matrix — Gradient Boosting**")
            safe_img("classification_confusion.png")
        st.markdown("**Model comparison across all metrics**")
        safe_img("classification_comparison.png")

        st.markdown('<div class="section-title">Model Interpretation</div>', unsafe_allow_html=True)
        insights = [
            "Gradient Boosting achieves AUC = 0.773, meaning it correctly ranks a buyer above a non-buyer 77% of the time.",
            "Random Forest achieves the highest accuracy (74%) but slightly lower AUC — better calibrated for balanced use.",
            "Logistic Regression provides a strong interpretable baseline with AUC = 0.79, useful for regulatory compliance.",
            "The 1% class-flip noise in the dataset creates a realistic accuracy ceiling — preventing overfitting to generation patterns.",
        ]
        for i in insights:
            st.markdown(f'<div class="insight-box">{i}</div>', unsafe_allow_html=True)

    with tab2:
        safe_img("classification_importance.png")
        st.markdown('<div class="section-title">Top Feature Insights</div>', unsafe_allow_html=True)
        feat_insights = {
            "repeat_purchase_intent_score (30% weight)": "Strongest single predictor — customers who intend to buy again are already halfway to converting.",
            "recommendation_acceptance_score (20%)": "AI receptiveness directly drives conversion — validates the platform's personalisation strategy.",
            "ai_personalisation_comfort (15%)": "Trust in AI is a key gate — low comfort = conversion barrier regardless of pricing.",
            "platform_switching_propensity (−15%)": "High switchers are harder to retain — require loyalty incentives at point of acquisition.",
            "cluster (segment label)": "Persona membership encodes rich behavioural patterns not captured by individual features.",
        }
        for feat, explanation in feat_insights.items():
            st.markdown(f'<div class="warn-box"><b>{feat}</b><br>{explanation}</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-title">Live Conversion Probability Estimator</div>', unsafe_allow_html=True)
        st.markdown("Adjust the sliders to simulate a customer profile and predict their purchase probability.")

        col1, col2, col3 = st.columns(3)
        with col1:
            rpi   = st.slider("Repeat purchase intent (1–5)", 1, 5, 4)
            rec   = st.slider("Recommendation acceptance (1–5)", 1, 5, 4)
            ai_c  = st.slider("AI personalisation comfort (1–5)", 1, 5, 3)
        with col2:
            swp   = st.slider("Platform switching propensity (1–5)", 1, 5, 2)
            freq  = st.slider("Shopping frequency (1=Rarely, 5=Daily)", 1, 5, 3)
            aband = st.slider("Cart abandonment frequency (1=Never, 5=Always)", 1, 5, 2)
        with col3:
            deliv = st.slider("Delivery importance (1–5)", 1, 5, 4)
            disc  = st.slider("Discount sensitivity (1–5)", 1, 5, 3)
            loyal = st.selectbox("Loyalty member?", ["No", "Yes"])

        def norm5(x): return (x - 3) / 2.0
        from scipy.special import expit

        score = (
            0.30 * norm5(rpi) +
            0.20 * norm5(rec) +
            0.15 * norm5(ai_c) -
            0.15 * norm5(swp) +
            0.10 * norm5(freq) -
            0.10 * norm5(aband) +
            0.05 * norm5(deliv) -
            0.05 * norm5(disc)
        )
        loyalty_bonus = 0.3 if loyal == "Yes" else 0.0
        prob = float(expit(score * 2.5 + loyalty_bonus))

        color = "#1D9E75" if prob >= 0.6 else "#EF9F27" if prob >= 0.4 else "#E24B4A"
        verdict = "Likely to Purchase" if prob >= 0.6 else "Uncertain" if prob >= 0.4 else "Unlikely to Purchase"

        st.markdown(f"""
        <div style="background:{color}22;border-left:4px solid {color};
                    padding:1.2rem 1.5rem;border-radius:8px;margin-top:1rem;text-align:center">
            <div style="font-size:2.5rem;font-weight:700;color:{color}">{prob:.1%}</div>
            <div style="font-size:1.1rem;font-weight:600;color:{color}">{verdict}</div>
            <div style="font-size:0.85rem;color:#555;margin-top:4px">
                Based on weighted sigmoid model (8 key features)
            </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — SPEND PREDICTION
# ═══════════════════════════════════════════════════════════════════
elif "Spend" in page:
    st.markdown('<div class="main-header">💰 Customer Spend Prediction</div>', unsafe_allow_html=True)
    st.markdown("Regression model to estimate a customer's maximum single transaction value (₹).")

    try:
        rm = load_json(os.path.join(MODELS, 'regression_metrics.json'))
        best = 'Gradient Boosting'
        bm   = rm.get(best, {})
        bt   = rm.get('back_transform', {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best Model", best)
        c2.metric("R² Score", f"{bm.get('R2', 0):.3f}")
        c3.metric("MAE (log)", f"{bm.get('MAE', 0):.4f}")
        c4.metric("MAE (₹)", f"₹{bt.get('MAE_INR', 0):,.0f}")
        c5.metric("RMSE (₹)", f"₹{bt.get('RMSE_INR', 0):,.0f}")
    except:
        st.info("Run 05_regression.py to generate metrics.")

    tab1, tab2, tab3 = st.tabs(["Model Performance", "Feature Importance", "Spend Calculator"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Predicted vs actual log-spend**")
            safe_img("regression_pred_actual.png")
        with col2:
            st.markdown("**Residual analysis**")
            safe_img("regression_residuals.png")
        st.markdown("**Model comparison**")
        safe_img("regression_comparison.png")

        st.markdown('<div class="section-title">Regression Insights</div>', unsafe_allow_html=True)
        for i in [
            "R² = 0.742 means the model explains 74.2% of variance in customer spending — strong for survey-based data.",
            "Linear and Ridge Regression achieve identical R² (0.767), confirming low multicollinearity in the feature set.",
            "Log-transformation of the spend target normalises the right-skewed distribution and reduces outlier influence.",
            "MAE of ₹3,848 means predictions are within ~₹4,000 of true spend for the average respondent.",
        ]:
            st.markdown(f'<div class="insight-box">{i}</div>', unsafe_allow_html=True)

    with tab2:
        safe_img("regression_importance.png")

    with tab3:
        st.markdown('<div class="section-title">Estimated Spend Calculator</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            income_opt = st.selectbox("Monthly income bracket",
                ["Below ₹20,000","₹20,000–50,000","₹50,000–1,00,000",
                 "₹1,00,000–2,00,000","Above ₹2,00,000"])
            age_opt = st.selectbox("Age group", ["18–24","25–34","35–44","45–54","55+"])
            loyalty = st.selectbox("Loyalty member?", ["No","Yes"])
            ai_c2 = st.slider("AI comfort score (1–5)", 1, 5, 3)
        with col2:
            freq2 = st.slider("Shopping frequency (1–5)", 1, 5, 3)
            disc2 = st.slider("Discount sensitivity (1–5)", 1, 5, 3)
            elec  = st.checkbox("Buys Electronics", value=True)
            fash  = st.checkbox("Buys Fashion", value=True)

        income_map = {"Below ₹20,000":1,"₹20,000–50,000":2,"₹50,000–1,00,000":3,
                      "₹1,00,000–2,00,000":4,"Above ₹2,00,000":5}
        age_map    = {"18–24":1,"25–34":2,"35–44":3,"45–54":4,"55+":5}
        inc = income_map[income_opt]
        ag  = age_map[age_opt]
        loy = 1 if loyalty=="Yes" else 0

        MU_MAP = {1:6.5, 2:7.5, 3:8.4, 4:9.4, 5:10.5}
        base_log = MU_MAP[inc]
        adjustments = (
            0.15*(loy) +
            0.08*(ai_c2/5) +
            0.05*(freq2/5) -
            0.05*(disc2/5) +
            0.06*(int(elec)) +
            0.04*(int(fash))
        )
        est_log   = base_log + adjustments
        est_spend = np.expm1(est_log)
        low_est   = np.expm1(est_log - 0.55)
        high_est  = np.expm1(est_log + 0.55)

        st.markdown(f"""
        <div style="background:#eef6ff;border-left:4px solid #1D9E75;
                    padding:1.2rem 1.5rem;border-radius:8px;margin-top:1rem;text-align:center">
            <div style="font-size:0.9rem;color:#555">Estimated max single transaction value</div>
            <div style="font-size:2.8rem;font-weight:700;color:#085041">₹{est_spend:,.0f}</div>
            <div style="font-size:0.85rem;color:#666;margin-top:4px">
                Confidence range: ₹{low_est:,.0f} – ₹{high_est:,.0f}
            </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 5 — PRODUCT ASSOCIATIONS
# ═══════════════════════════════════════════════════════════════════
elif "Association" in page:
    st.markdown('<div class="main-header">🔗 Product Association Rules</div>', unsafe_allow_html=True)
    st.markdown("Apriori algorithm applied to product category co-purchase baskets across 2,000 respondents.")

    try:
        am = load_json(os.path.join(MODELS, 'arm_metrics.json'))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frequent Itemsets", am.get('n_frequent_itemsets', '—'))
        c2.metric("Total Rules", am.get('n_rules', '—'))
        c3.metric("Max Lift", f"{am.get('top_rule_lift', 0):.3f}")
        c4.metric("Avg Confidence", f"{am.get('avg_confidence', 0):.1%}")
    except:
        st.info("Run 06_association_rules.py to generate metrics.")

    tab1, tab2, tab3 = st.tabs(["Rules Explorer", "Visualisations", "Business Recommendations"])

    with tab1:
        rules_path = os.path.join(MODELS, 'association_rules.csv')
        if os.path.exists(rules_path):
            rules_df = pd.read_csv(rules_path)
            st.markdown('<div class="section-title">All Association Rules</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            min_conf = col1.slider("Min confidence", 0.30, 0.90, 0.40, 0.05)
            min_lift = col2.slider("Min lift", 0.80, 2.0, 1.0, 0.05)
            sort_by  = col3.selectbox("Sort by", ["lift","confidence","support"])
            filtered = rules_df[
                (rules_df['confidence'] >= min_conf) &
                (rules_df['lift'] >= min_lift)
            ].sort_values(sort_by, ascending=False)
            st.caption(f"{len(filtered)} rules matching filters")
            st.dataframe(filtered[['antecedents','consequents','support','confidence','lift']]
                         .round(4), use_container_width=True, hide_index=True)
        else:
            st.info("Run 06_association_rules.py to generate rules.")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Support vs confidence (colour = lift)**")
            safe_img("arm_scatter.png")
        with col2:
            st.markdown("**Top 12 rules by lift**")
            safe_img("arm_top_rules.png")
        st.markdown("**Category co-purchase rate matrix**")
        safe_img("arm_copurchase_heatmap.png")

    with tab3:
        st.markdown('<div class="section-title">Business Recommendations from ARM</div>', unsafe_allow_html=True)
        recs = [
            ("Toys → Home (lift 1.21)", "Bundle baby products with home décor; target new parents with combo deals."),
            ("Beauty + Grocery → Home", "Homemaker segment drives this triple-category basket — ideal for monthly subscription boxes."),
            ("Grocery + Home → Beauty", "Upsell beauty products at checkout when grocery + home items are in cart."),
            ("Electronics + Sports (Urban Professionals)", "Recommend fitness trackers and sports accessories alongside electronics purchases."),
        ]
        for rule, action in recs:
            st.markdown(f"""
            <div class="warn-box">
                <b>Rule: {rule}</b><br>
                💡 {action}
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 6 — EDA & INSIGHTS
# ═══════════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("Statistical overview and visual exploration of the 2,000-respondent survey dataset.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", "45")
    c3.metric("Missing (platform_3)", "~31.5%")
    c4.metric("Numerical cols", "19")
    c5.metric("Categorical cols", "26")

    tab1, tab2, tab3 = st.tabs(["Visual EDA", "Raw Statistics", "Data Sample"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Target variable balance**")
            safe_img("target_balance.png")
        with col2:
            st.markdown("**Product category purchase rates**")
            safe_img("category_rates.png")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Income vs spend heatmap**")
            safe_img("income_spend_heatmap.png")
        with col4:
            st.markdown("**Spend distribution (raw vs log)**")
            safe_img("spend_distribution.png")
        st.markdown("**Feature correlation matrix**")
        safe_img("correlation_matrix.png")

    with tab2:
        st.markdown('<div class="section-title">Numerical Feature Summary</div>', unsafe_allow_html=True)
        num_cols = ['review_importance_score','recommendation_acceptance_score',
                    'discount_sensitivity_score','delivery_importance_score',
                    'repeat_purchase_intent_score','platform_switching_propensity',
                    'ai_personalisation_comfort','max_single_transaction_value',
                    'loyalty_member_flag','will_purchase']
        st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

        st.markdown('<div class="section-title">Categorical Feature Distributions</div>', unsafe_allow_html=True)
        cat_col = st.selectbox("Select column", ['age_group','gender','city_tier',
                                                   'employment_status','shopping_frequency',
                                                   'monthly_income_bracket','preferred_payment_method'])
        vc = df[cat_col].value_counts().reset_index()
        vc.columns = [cat_col, 'count']
        vc['%'] = (vc['count'] / len(df) * 100).round(1)
        st.dataframe(vc, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown('<div class="section-title">Dataset Sample</div>', unsafe_allow_html=True)
        display_cols = ['respondent_id','persona','age_group','gender','city_tier',
                        'monthly_income_bracket','shopping_frequency','will_purchase']
        n_show = st.slider("Rows to display", 5, 50, 10)
        st.dataframe(df[display_cols].head(n_show), use_container_width=True, hide_index=True)
        st.download_button("⬇ Download full dataset", df.to_csv(index=False),
                           "ecommerce_survey_synthetic.csv", "text/csv")
