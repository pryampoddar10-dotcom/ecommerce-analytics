"""
06_association_rules.py
Product Category Association Rule Mining — Apriori Algorithm
Author: Pryam Poddar | SP Jain School of Global Management | MGB Term 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os, warnings, json
warnings.filterwarnings('ignore')

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, 'dataset', 'ecommerce_clustered.csv')
PLOTS  = os.path.join(BASE, 'dashboard', 'plots')
MODELS = os.path.join(BASE, 'dashboard', 'models')

df = pd.read_csv(DATA)
print(f"Loaded: {df.shape}")

# ── Build transaction baskets from category columns ───────────────
CAT_COLS = ['cat_fashion','cat_electronics','cat_grocery','cat_beauty',
            'cat_home','cat_books','cat_sports','cat_toys']
CAT_NAMES = ['Fashion','Electronics','Grocery','Beauty',
             'Home','Books','Sports','Toys']

# Convert binary matrix → list of items per transaction
transactions = []
for _, row in df[CAT_COLS].iterrows():
    basket = [CAT_NAMES[i] for i, v in enumerate(row) if v == 1]
    if basket:
        transactions.append(basket)

print(f"Transactions (non-empty baskets): {len(transactions)}")
print(f"Average basket size: {np.mean([len(t) for t in transactions]):.2f}")

# ── Encode & run Apriori ──────────────────────────────────────────
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
basket_df = pd.DataFrame(te_array, columns=te.columns_)

freq_items = apriori(basket_df, min_support=0.10, use_colnames=True, max_len=3)
freq_items['length'] = freq_items['itemsets'].apply(len)
print(f"\nFrequent itemsets (support≥0.10): {len(freq_items)}")

rules = association_rules(freq_items, metric='confidence', min_threshold=0.40)
rules = rules.sort_values('lift', ascending=False)
print(f"Association rules (confidence≥0.40): {len(rules)}")
print(f"\nTop 10 rules by lift:")
print(rules[['antecedents','consequents','support','confidence','lift']].head(10).to_string())

# ── Per-persona rules ─────────────────────────────────────────────
persona_rules = {}
for persona in df['cluster_name'].dropna().unique():
    sub = df[df['cluster_name'] == persona]
    trans_p = []
    for _, row in sub[CAT_COLS].iterrows():
        basket = [CAT_NAMES[i] for i, v in enumerate(row) if v == 1]
        if basket:
            trans_p.append(basket)
    if len(trans_p) < 30:
        continue
    te_p = TransactionEncoder()
    ta_p = te_p.fit_transform(trans_p)
    bd_p = pd.DataFrame(ta_p, columns=te_p.columns_)
    fi_p = apriori(bd_p, min_support=0.12, use_colnames=True, max_len=2)
    if len(fi_p) == 0:
        continue
    rp = association_rules(fi_p, metric='confidence', min_threshold=0.35)
    if len(rp) > 0:
        rp = rp.sort_values('lift', ascending=False).head(5)
        persona_rules[persona] = rp[['antecedents','consequents',
                                      'support','confidence','lift']].copy()
        print(f"\n--- {persona} ---")
        print(rp[['antecedents','consequents','support','confidence','lift']].head(5).to_string())

# ── Plot 1: Support vs Confidence scatter ────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(rules['support'], rules['confidence'],
                     c=rules['lift'], cmap='YlOrRd', s=60, alpha=0.7, edgecolors='none')
plt.colorbar(scatter, ax=ax, label='Lift')
ax.set_xlabel('Support')
ax.set_ylabel('Confidence')
ax.set_title('Association rules — support vs confidence (colour = lift)')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'arm_scatter.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Plot 2: Top rules by lift ────────────────────────────────────
top_rules = rules.head(12).copy()
top_rules['rule'] = (top_rules['antecedents'].apply(lambda x: '+'.join(sorted(x)))
                     + ' → ' +
                     top_rules['consequents'].apply(lambda x: '+'.join(sorted(x))))
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(top_rules['rule'][::-1], top_rules['lift'][::-1], color='#D85A30')
for bar, v in zip(bars, top_rules['lift'][::-1]):
    ax.text(v+0.01, bar.get_y()+bar.get_height()/2, f'{v:.2f}',
            va='center', fontsize=9)
ax.set_xlabel('Lift')
ax.set_title('Top 12 association rules by lift')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'arm_top_rules.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Plot 3: Category co-purchase heatmap ─────────────────────────
co_matrix = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        if i != j:
            co_matrix[i,j] = (df[CAT_COLS[i]] & df[CAT_COLS[j]]).mean()
        else:
            co_matrix[i,j] = df[CAT_COLS[i]].mean()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(co_matrix, cmap='Greens', vmin=0, vmax=0.5)
ax.set_xticks(range(8)); ax.set_yticks(range(8))
ax.set_xticklabels(CAT_NAMES, rotation=45, ha='right')
ax.set_yticklabels(CAT_NAMES)
for i in range(8):
    for j in range(8):
        ax.text(j, i, f'{co_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)
ax.set_title('Product category co-purchase rates')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'arm_copurchase_heatmap.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Save rules ────────────────────────────────────────────────────
rules_save = rules.copy()
rules_save['antecedents'] = rules_save['antecedents'].apply(lambda x: ', '.join(sorted(x)))
rules_save['consequents'] = rules_save['consequents'].apply(lambda x: ', '.join(sorted(x)))
rules_save = rules_save.round(4)
rules_save.to_csv(os.path.join(MODELS, 'association_rules.csv'), index=False)

summary = {
    'n_frequent_itemsets': int(len(freq_items)),
    'n_rules': int(len(rules)),
    'top_rule_lift': float(rules['lift'].max()),
    'avg_confidence': float(rules['confidence'].mean()),
    'avg_support': float(rules['support'].mean()),
}
with open(os.path.join(MODELS, 'arm_metrics.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ ARM complete. {len(rules)} rules | Max lift={rules['lift'].max():.3f}")
