"""
04_classification.py
Purchase Conversion Prediction — Random Forest + XGBoost
Author: Pryam Poddar | SP Jain School of Global Management | MGB Term 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os, warnings, pickle, json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score, accuracy_score)

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, 'dataset', 'ecommerce_clustered.csv')
PLOTS  = os.path.join(BASE, 'dashboard', 'plots')
MODELS = os.path.join(BASE, 'dashboard', 'models')

df = pd.read_csv(DATA)
print(f"Loaded: {df.shape}")

# ── Features ─────────────────────────────────────────────────────
FEATURES = [
    'age_enc','city_tier_enc','income_enc','spend_enc','education_enc',
    'shopping_freq_enc','browse_time_enc','abandon_freq_enc',
    'return_freq_enc','social_inf_enc','discount_thresh_enc','festive_enc',
    'review_importance_score','recommendation_acceptance_score',
    'discount_sensitivity_score','delivery_importance_score',
    'repeat_purchase_intent_score','platform_switching_propensity',
    'ai_personalisation_comfort','loyalty_member_flag',
    'cat_fashion','cat_electronics','cat_grocery','cat_beauty',
    'cat_home','cat_books','cat_sports','cat_toys','cluster'
]

TARGET = 'will_purchase'
X = df[FEATURES].fillna(0)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Class balance (train): {y_train.value_counts(normalize=True).round(3).to_dict()}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Models ───────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                   min_samples_leaf=10, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                       learning_rate=0.1, random_state=42),
}

results = {}
for name, model in models.items():
    X_tr = X_train_s if name == 'Logistic Regression' else X_train.values
    X_te = X_test_s  if name == 'Logistic Regression' else X_test.values
    model.fit(X_tr, y_train)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    f1   = f1_score(y_test, y_pred)
    X_full_cv = X_train_s if name == 'Logistic Regression' else X_train.values
    cv   = cross_val_score(model, X_full_cv,
                           y_train, cv=5, scoring='roc_auc').mean()
    results[name] = {'accuracy': round(acc,4), 'roc_auc': round(auc,4),
                     'f1': round(f1,4), 'cv_auc': round(cv,4),
                     'y_pred': y_pred, 'y_proba': y_proba}
    print(f"\n{name}: Acc={acc:.4f} | AUC={auc:.4f} | F1={f1:.4f} | CV-AUC={cv:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No Purchase','Purchase']))

# ── Best model = Gradient Boosting ───────────────────────────────
BEST = 'Gradient Boosting'
best_model = models[BEST]
best_res   = results[BEST]

# ── Plot 1: ROC curves ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
COLORS = {'Logistic Regression':'#EF9F27','Random Forest':'#1D9E75','Gradient Boosting':'#7F77DD'}
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})",
            color=COLORS[name], linewidth=2)
ax.plot([0,1],[0,1],'--', color='gray', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curves — purchase conversion classifiers')
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'classification_roc.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Plot 2: Confusion matrix ─────────────────────────────────────
cm = confusion_matrix(y_test, best_res['y_pred'])
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Purples')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Predicted No','Predicted Yes'])
ax.set_yticklabels(['Actual No','Actual Yes'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                fontsize=16, fontweight='bold',
                color='white' if cm[i,j] > cm.max()/2 else 'black')
ax.set_title(f'Confusion matrix — {BEST}')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'classification_confusion.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Plot 3: Feature importance ───────────────────────────────────
fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
top15 = fi.tail(15)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(top15.index, top15.values, color='#7F77DD')
ax.set_xlabel('Feature importance (Gini)')
ax.set_title(f'Top 15 features — {BEST}')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'classification_importance.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Plot 4: Model comparison bar ─────────────────────────────────
metrics_df = pd.DataFrame({n: {'Accuracy': r['accuracy'], 'ROC-AUC': r['roc_auc'],
                                'F1-Score': r['f1']} for n, r in results.items()}).T
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(metrics_df))
w = 0.25
for i, (col, color) in enumerate(zip(['Accuracy','ROC-AUC','F1-Score'],
                                      ['#7F77DD','#1D9E75','#EF9F27'])):
    ax.bar(x + i*w, metrics_df[col], w, label=col, color=color)
ax.set_xticks(x + w)
ax.set_xticklabels(metrics_df.index, fontsize=10)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel('Score')
ax.set_title('Model comparison — classification metrics')
ax.legend()
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'classification_comparison.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Save best model + metrics ─────────────────────────────────────
with open(os.path.join(MODELS, 'classifier.pkl'), 'wb') as f:
    pickle.dump({'model': best_model, 'features': FEATURES, 'name': BEST}, f)

save_metrics = {n: {k: v for k, v in r.items() if k not in ['y_pred','y_proba']}
                for n, r in results.items()}
with open(os.path.join(MODELS, 'classification_metrics.json'), 'w') as f:
    json.dump(save_metrics, f, indent=2)

print(f"\n✓ Classification complete. Best model: {BEST} | AUC={best_res['roc_auc']:.4f}")
