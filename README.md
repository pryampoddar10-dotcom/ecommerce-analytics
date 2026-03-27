# AI-Driven E-Commerce Analytics Platform

**Author:** Pryam Poddar  
**Programme:** Master of Global Business (MGB) — Term 2  
**Institution:** SP Jain School of Global Management  
**Subject:** Data Analytics (Individual Assignment)

---

## Project Overview

This project demonstrates how an **AI-driven e-commerce platform** can use data analytics and machine learning to optimise customer targeting, dynamic pricing, and purchase conversion in the Indian market.

A synthetic survey dataset of **2,000 Indian consumers** was designed, generated, and analysed using four core ML techniques.

---

## Live Dashboard

🔗 **[View on Streamlit Cloud](https://your-app-name.streamlit.app)**  
*(Replace with your deployed URL after following deployment steps below)*

---

## Repository Structure

```
├── dataset/
│   ├── ecommerce_survey_synthetic.csv   # 2,000 rows · 45 columns
│   ├── ecommerce_ml_ready.csv           # Encoded + feature-engineered
│   └── ecommerce_clustered.csv          # With cluster labels
│
├── code/
│   ├── 01_data_generation.py            # Synthetic dataset generation
│   ├── 02_eda_preprocessing.py          # EDA + encoding + plots
│   ├── 03_clustering.py                 # K-Means segmentation
│   ├── 04_classification.py             # Purchase conversion classifier
│   ├── 05_regression.py                 # Spend prediction regressor
│   ├── 06_association_rules.py          # Apriori ARM
│   └── apriori_utils.py                 # Lightweight Apriori (no mlxtend needed)
│
├── dashboard/
│   ├── app.py                           # Streamlit multi-page dashboard
│   ├── requirements.txt                 # Python dependencies
│   ├── plots/                           # All generated visualisation PNGs
│   └── models/                          # Saved model PKLs + metric JSONs
│
├── report/
│   └── Pryam_Poddar_DA_Report.pdf       # Full academic report
│
└── README.md
```

---

## ML Techniques Applied

| Technique | Algorithm | Target | Key Metric |
|-----------|-----------|--------|------------|
| Clustering | K-Means (k=5) | Customer segments | Silhouette = 0.078 |
| Classification | Gradient Boosting | `will_purchase` (binary) | AUC = 0.773 |
| Regression | Gradient Boosting | `log_spend` → ₹ | R² = 0.742 |
| Association Rules | Apriori | Category co-purchase | 131 rules · Max lift = 1.21 |

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-analytics.git
cd ecommerce-analytics
```

### 2. Install dependencies
```bash
pip install -r dashboard/requirements.txt
```

### 3. Run ML pipeline (in order)
```bash
python code/02_eda_preprocessing.py
python code/03_clustering.py
python code/04_classification.py
python code/05_regression.py
python code/06_association_rules.py
```

### 4. Launch dashboard
```bash
streamlit run dashboard/app.py
```

---

## Deploy to Streamlit Cloud

1. Push this repository to GitHub (public repo)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo · Branch: `main` · Main file: `dashboard/app.py`
4. Click **Deploy** — live in ~2 minutes

> **Important:** Streamlit Cloud runs `dashboard/app.py` with `dashboard/` as the working directory.  
> The app uses relative paths (`../dataset/`, `models/`, `plots/`) — these resolve correctly when deployed.

---

## Dataset Description

The synthetic dataset simulates 2,000 Indian e-commerce consumers across 5 archetypes:

| Persona | Share | Description |
|---------|-------|-------------|
| Urban Professional | 30% | Metro · High income · AI-receptive · Weekly shopper |
| Budget Student | 22% | Age 18–24 · Low income · Discount-driven |
| Value Seeker | 20% | Tier 2/3 · COD preferred · Price-comparing |
| Homemaker | 16% | Fashion + Grocery basket · Social-influenced |
| Premium Loyalist | 12% | Highest spend · Multi-loyalty · Daily shopper |

**Target variable:** `will_purchase` (binary) — 61.8% positive class

---

## Key Business Insights

- Repeat purchase intent is the **#1 predictor** of conversion (weight: 30%)
- AI personalisation comfort is a **conversion gate** — low comfort = lower purchase rate
- Toys → Home has the **highest lift rule** (1.21) — actionable for product bundling
- Premium Loyalists (12%) generate disproportionate revenue — priority segment for retention
- Discount sensitivity is **inversely correlated** with income (r = −0.60)

---

## Contact

**Pryam Poddar**  
MGB Term 2, SP Jain School of Global Management  
📧 pryam.poddar@spjain.org
