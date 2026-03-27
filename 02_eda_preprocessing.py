"""
02_eda_preprocessing.py
EDA and preprocessing for the AI-Driven E-Commerce Analytics Project
Author: Pryam Poddar | SP Jain School of Global Management | MGB Term 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, 'dataset', 'ecommerce_survey_synthetic.csv')
OUT    = os.path.join(BASE, 'dataset')
PLOTS  = os.path.join(BASE, 'dashboard', 'plots')
os.makedirs(PLOTS, exist_ok=True)

PALETTE = ['#7F77DD','#1D9E75','#EF9F27','#D85A30','#D4537E']

# ── 1. Load ───────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}\n")

# ── 2. Basic stats ────────────────────────────────────────────────
print("=== NUMERICAL SUMMARY ===")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[num_cols].describe().round(2).to_string())

# ── 3. Ordinal encoding maps ─────────────────────────────────────
AGE_MAP       = {'18-24':1,'25-34':2,'35-44':3,'45-54':4,'55+':5}
FREQ_MAP      = {'Rarely':1,'Occasionally':2,'Monthly':3,'Weekly':4,'Daily or almost daily':5}
BROWSE_MAP    = {'Under 10 minutes':1,'10-30 minutes':2,'30-60 minutes':3,'More than 1 hour':4}
ABANDON_MAP   = {'Never':1,'Rarely':2,'Sometimes':3,'Often':4,'Almost always':5}
EDU_MAP       = {'High school or below':1,'Undergraduate':2,'Postgraduate':3,'Doctoral':4}
CITY_MAP      = {'Metro (Tier 1)':4,'Tier 2 city':3,'Tier 3 city':2,'Rural / town':1}
INCOME_MAP    = {'Below 20000':1,'20000-50000':2,'50000-100000':3,'100000-200000':4,'Above 200000':5}
SPEND_MAP     = {'Under 500':1,'500-2000':2,'2000-5000':3,'5000-15000':4,'Above 15000':5}
RETURN_MAP    = {'Never':1,'Rarely':2,'Occasionally':3,'Frequently':4}
SOCIAL_INF_MAP= {'Never':1,'Rarely':2,'Sometimes':3,'Often':4,'Almost always':5}
THRESH_MAP    = {'Discounts dont trigger':1,'10-20pct':2,'21-40pct':3,'41-60pct':4,'Above 60pct':5}
FESTIVE_MAP   = {'Less than 10pct':1,'10-25pct':2,'26-50pct':3,'More than 50pct':4}

df['age_enc']             = df['age_group'].map(AGE_MAP)
df['shopping_freq_enc']   = df['shopping_frequency'].map(FREQ_MAP)
df['browse_time_enc']     = df['browse_time_before_purchase'].map(BROWSE_MAP)
df['abandon_freq_enc']    = df['cart_abandonment_freq'].map(ABANDON_MAP)
df['education_enc']       = df['education_level'].map(EDU_MAP)
df['city_tier_enc']       = df['city_tier'].map(CITY_MAP)
df['income_enc']          = df['monthly_income_bracket'].map(INCOME_MAP)
df['spend_enc']           = df['monthly_spend_bracket'].map(SPEND_MAP)
df['return_freq_enc']     = df['return_frequency'].map(RETURN_MAP)
df['social_inf_enc']      = df['social_purchase_influence_freq'].map(SOCIAL_INF_MAP)
df['discount_thresh_enc'] = df['impulse_discount_threshold'].map(THRESH_MAP)
df['festive_enc']         = df['festive_spend_proportion'].map(FESTIVE_MAP)

# One-hot encode nominal categoricals
nom_cols = ['gender','city_tier','employment_status','primary_device',
            'discovery_channel','preferred_payment_method',
            'social_commerce_adoption','impulse_trigger_type',
            'cart_abandon_reason']
df_ohe = pd.get_dummies(df[nom_cols], drop_first=False, dtype=int)

# ── 4. Outlier treatment (Winsorise spend at 99th pct) ──────────
p99 = df['max_single_transaction_value'].quantile(0.99)
df['spend_winsorised'] = df['max_single_transaction_value'].clip(upper=p99)
df['log_spend']        = np.log1p(df['spend_winsorised'])
print(f"\nSpend winsorised at 99th pct: ₹{p99:,.0f}")
print(f"Log-spend: mean={df['log_spend'].mean():.3f}, std={df['log_spend'].std():.3f}")

# ── 5. Build final ML-ready dataframe ────────────────────────────
feature_cols = (
    ['age_enc','shopping_freq_enc','browse_time_enc','abandon_freq_enc',
     'education_enc','city_tier_enc','income_enc','spend_enc',
     'return_freq_enc','social_inf_enc','discount_thresh_enc','festive_enc',
     'review_importance_score','recommendation_acceptance_score',
     'discount_sensitivity_score','delivery_importance_score',
     'repeat_purchase_intent_score','platform_switching_propensity',
     'ai_personalisation_comfort','loyalty_member_flag',
     'cat_fashion','cat_electronics','cat_grocery','cat_beauty',
     'cat_home','cat_books','cat_sports','cat_toys',
     'log_spend'] +
    list(df_ohe.columns)
)

df_ml = pd.concat([df.reset_index(drop=True), df_ohe.reset_index(drop=True)], axis=1)
df_ml = df_ml.dropna(subset=['age_enc','income_enc'])

df_ml.to_csv(os.path.join(OUT, 'ecommerce_ml_ready.csv'), index=False)
print(f"\nML-ready dataset: {df_ml.shape} → saved to dataset/ecommerce_ml_ready.csv")

# ── 6. EDA Plots ─────────────────────────────────────────────────

# Plot 1: Persona distribution
fig, ax = plt.subplots(figsize=(8,4))
counts = df['persona'].value_counts()
bars = ax.barh(counts.index, counts.values, color=PALETTE)
for bar, v in zip(bars, counts.values):
    ax.text(v+8, bar.get_y()+bar.get_height()/2, str(v), va='center', fontsize=10)
ax.set_xlabel('Count')
ax.set_title('Respondent distribution by customer persona')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'persona_distribution.png'), dpi=120, bbox_inches='tight')
plt.close()

# Plot 2: Target balance
fig, axes = plt.subplots(1, 2, figsize=(10,4))
vc = df['will_purchase'].value_counts()
axes[0].pie(vc.values, labels=['Will purchase (1)','Will not (0)'],
            colors=['#1D9E75','#E24B4A'], autopct='%1.1f%%', startangle=90)
axes[0].set_title('Target variable balance')
purchase_by_persona = df.groupby('persona')['will_purchase'].mean().sort_values()
axes[1].barh(purchase_by_persona.index, purchase_by_persona.values,
             color=PALETTE[:len(purchase_by_persona)])
axes[1].set_xlabel('Purchase rate')
axes[1].set_title('Purchase rate by persona')
axes[1].spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'target_balance.png'), dpi=120, bbox_inches='tight')
plt.close()

# Plot 3: Income vs Spend heatmap proxy
fig, ax = plt.subplots(figsize=(8,5))
ct = pd.crosstab(df['monthly_income_bracket'], df['monthly_spend_bracket'])
income_order = ['Below 20000','20000-50000','50000-100000','100000-200000','Above 200000']
spend_order  = ['Under 500','500-2000','2000-5000','5000-15000','Above 15000']
ct = ct.reindex(index=[i for i in income_order if i in ct.index],
                columns=[s for s in spend_order if s in ct.columns], fill_value=0)
sns.heatmap(ct, annot=True, fmt='d', cmap='Purples', ax=ax, linewidths=0.5)
ax.set_title('Income bracket vs monthly spend bracket')
ax.set_xlabel('Monthly spend')
ax.set_ylabel('Monthly income')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'income_spend_heatmap.png'), dpi=120, bbox_inches='tight')
plt.close()

# Plot 4: Category purchase rates
cats = ['cat_fashion','cat_electronics','cat_grocery','cat_beauty',
        'cat_home','cat_books','cat_sports','cat_toys']
cat_labels = ['Fashion','Electronics','Grocery','Beauty','Home','Books','Sports','Toys']
cat_rates  = [df[c].mean() for c in cats]
fig, ax = plt.subplots(figsize=(8,4))
bars = ax.bar(cat_labels, cat_rates, color=PALETTE*2)
for bar, v in zip(bars, cat_rates):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f'{v:.1%}',
            ha='center', fontsize=9)
ax.set_ylabel('Purchase rate')
ax.set_title('Product category purchase rates across all respondents')
ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0, max(cat_rates)+0.1)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'category_rates.png'), dpi=120, bbox_inches='tight')
plt.close()

# Plot 5: Spend distribution (log scale)
fig, axes = plt.subplots(1, 2, figsize=(11,4))
axes[0].hist(df['max_single_transaction_value'], bins=50,
             color='#7F77DD', edgecolor='white', linewidth=0.3)
axes[0].set_title('Max single transaction value (raw)')
axes[0].set_xlabel('₹')
axes[0].spines[['top','right']].set_visible(False)
axes[1].hist(df['log_spend'], bins=40, color='#1D9E75', edgecolor='white', linewidth=0.3)
axes[1].set_title('Log-transformed spend (normalised)')
axes[1].set_xlabel('ln(₹)')
axes[1].spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'spend_distribution.png'), dpi=120, bbox_inches='tight')
plt.close()

# Plot 6: Correlation heatmap of key numerical features
key_num = ['age_enc','income_enc','spend_enc','discount_sensitivity_score',
           'repeat_purchase_intent_score','ai_personalisation_comfort',
           'platform_switching_propensity','recommendation_acceptance_score',
           'shopping_freq_enc','will_purchase']
corr = df_ml[key_num].corr()
fig, ax = plt.subplots(figsize=(9,7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax, linewidths=0.5, annot_kws={'size':8})
ax.set_title('Correlation matrix — key analytical features')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'correlation_matrix.png'), dpi=120, bbox_inches='tight')
plt.close()

print("\n6 EDA plots saved to dashboard/plots/")
print("\n✓ Preprocessing complete.")
