"""
03_clustering.py
Customer Segmentation using K-Means Clustering
Author: Pryam Poddar | SP Jain School of Global Management | MGB Term 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os, warnings, pickle
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, 'dataset', 'ecommerce_ml_ready.csv')
PLOTS  = os.path.join(BASE, 'dashboard', 'plots')
MODELS = os.path.join(BASE, 'dashboard', 'models')
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

PALETTE = ['#7F77DD','#1D9E75','#EF9F27','#D85A30','#D4537E']

df = pd.read_csv(DATA)
print(f"Loaded: {df.shape}")

# ── Feature selection for clustering ────────────────────────────
CLUSTER_FEATURES = [
    'age_enc', 'city_tier_enc', 'income_enc', 'spend_enc',
    'shopping_freq_enc', 'discount_sensitivity_score',
    'repeat_purchase_intent_score', 'ai_personalisation_comfort',
    'loyalty_member_flag', 'recommendation_acceptance_score',
    'platform_switching_propensity', 'cat_fashion', 'cat_electronics',
    'cat_grocery', 'cat_beauty', 'cat_home', 'cat_books',
    'cat_sports', 'cat_toys', 'social_inf_enc',
]

X = df[CLUSTER_FEATURES].fillna(df[CLUSTER_FEATURES].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Elbow + Silhouette method ────────────────────────────────────
inertias, sil_scores = [], []
K_RANGE = range(2, 9)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels, sample_size=500))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(K_RANGE), inertias, 'o-', color='#7F77DD', linewidth=2)
axes[0].axvline(x=5, color='#E24B4A', linestyle='--', alpha=0.7, label='Chosen k=5')
axes[0].set_xlabel('Number of clusters (k)')
axes[0].set_ylabel('Inertia (WCSS)')
axes[0].set_title('Elbow method — optimal k selection')
axes[0].legend()
axes[0].spines[['top','right']].set_visible(False)

axes[1].plot(list(K_RANGE), sil_scores, 's-', color='#1D9E75', linewidth=2)
axes[1].axvline(x=5, color='#E24B4A', linestyle='--', alpha=0.7, label='Chosen k=5')
axes[1].set_xlabel('Number of clusters (k)')
axes[1].set_ylabel('Silhouette score')
axes[1].set_title('Silhouette score by k')
axes[1].legend()
axes[1].spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'clustering_elbow.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Final model: k=5 ────────────────────────────────────────────
K = 5
kmeans = KMeans(n_clusters=K, random_state=42, n_init=15)
df['cluster'] = kmeans.fit_predict(X_scaled)

sil  = silhouette_score(X_scaled, df['cluster'], sample_size=500)
db   = davies_bouldin_score(X_scaled, df['cluster'])
print(f"\nK-Means (k={K}): Silhouette={sil:.4f} | Davies-Bouldin={db:.4f}")

# ── Cluster profiling ────────────────────────────────────────────
PROFILE_COLS = ['income_enc','spend_enc','shopping_freq_enc',
                'discount_sensitivity_score','repeat_purchase_intent_score',
                'ai_personalisation_comfort','loyalty_member_flag',
                'platform_switching_propensity']
profile = df.groupby('cluster')[PROFILE_COLS].mean().round(2)

CLUSTER_NAMES = {
    profile['income_enc'].idxmax():        'Premium loyalists',
    profile['discount_sensitivity_score'].idxmax(): 'Deal hunters',
    profile['ai_personalisation_comfort'].idxmax(): 'Tech-savvy explorers',
    profile['shopping_freq_enc'].idxmin(): 'Occasional browsers',
}
remaining = [c for c in range(K) if c not in CLUSTER_NAMES]
CLUSTER_NAMES[remaining[0]] = 'Value seekers'
df['cluster_name'] = df['cluster'].map(CLUSTER_NAMES)
print(f"\nCluster name mapping: {CLUSTER_NAMES}")
print(f"\nCluster sizes:\n{df['cluster_name'].value_counts()}")

# ── PCA visualisation ────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(9, 6))
for i, (cid, cname) in enumerate(CLUSTER_NAMES.items()):
    mask = df['cluster'] == cid
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=PALETTE[i], label=cname, alpha=0.55, s=18, edgecolors='none')
centers_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
           c='black', s=120, marker='X', zorder=5, label='Centroids')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.set_title('Customer segments — PCA projection (K-Means, k=5)')
ax.legend(loc='upper right', fontsize=9)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'clustering_pca.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Radar / spider chart for cluster profiles ────────────────────
RADAR_COLS = ['income_enc','spend_enc','shopping_freq_enc',
              'repeat_purchase_intent_score','ai_personalisation_comfort',
              'loyalty_member_flag']
RADAR_LABELS = ['Income','Spend','Frequency','Repeat intent','AI comfort','Loyalty']
profile_radar = df.groupby('cluster')[RADAR_COLS].mean()
# normalise 0–1
profile_norm = (profile_radar - profile_radar.min()) / (profile_radar.max() - profile_radar.min() + 1e-9)

N_cats = len(RADAR_LABELS)
angles = np.linspace(0, 2*np.pi, N_cats, endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, K, figsize=(16, 3.5), subplot_kw=dict(polar=True))
for i, (cid, cname) in enumerate(CLUSTER_NAMES.items()):
    vals = profile_norm.loc[cid].tolist() + profile_norm.loc[cid].tolist()[:1]
    axes[i].plot(angles, vals, color=PALETTE[i], linewidth=2)
    axes[i].fill(angles, vals, color=PALETTE[i], alpha=0.25)
    axes[i].set_xticks(angles[:-1])
    axes[i].set_xticklabels(RADAR_LABELS, size=7)
    axes[i].set_yticks([])
    axes[i].set_title(cname, size=9, pad=12)
plt.suptitle('Cluster profiles — normalised feature radar', y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'clustering_radar.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Save model + labelled data ───────────────────────────────────
with open(os.path.join(MODELS, 'kmeans_model.pkl'), 'wb') as f:
    pickle.dump({'model': kmeans, 'scaler': scaler,
                 'features': CLUSTER_FEATURES, 'names': CLUSTER_NAMES}, f)

df.to_csv(os.path.join(BASE, 'dataset', 'ecommerce_clustered.csv'), index=False)

metrics = {'silhouette': round(sil,4), 'davies_bouldin': round(db,4),
           'k': K, 'cluster_names': CLUSTER_NAMES,
           'cluster_sizes': df['cluster_name'].value_counts().to_dict()}
pd.Series(metrics).to_json(os.path.join(MODELS, 'clustering_metrics.json'))

print(f"\n✓ Clustering complete. Silhouette={sil:.4f}")
