"""
05_regression.py
Customer Spend Prediction — Linear, Ridge, Random Forest Regressor
Author: Pryam Poddar | SP Jain School of Global Management | MGB Term 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os, warnings, pickle, json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, 'dataset', 'ecommerce_clustered.csv')
PLOTS  = os.path.join(BASE, 'dashboard', 'plots')
MODELS = os.path.join(BASE, 'dashboard', 'models')

df = pd.read_csv(DATA)
print(f"Loaded: {df.shape}")

# ── Features & target ────────────────────────────────────────────
REG_FEATURES = [
    'age_enc','city_tier_enc','income_enc','education_enc',
    'shopping_freq_enc','discount_sensitivity_score',
    'loyalty_member_flag','repeat_purchase_intent_score',
    'ai_personalisation_comfort','recommendation_acceptance_score',
    'cat_fashion','cat_electronics','cat_grocery','cat_beauty',
    'cat_home','cat_books','cat_sports','cat_toys',
    'delivery_importance_score','social_inf_enc','cluster'
]

TARGET = 'log_spend'   # predict log(spend), back-transform for interpretation

X = df[REG_FEATURES].fillna(0)
y = df[TARGET].fillna(df[TARGET].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Models ───────────────────────────────────────────────────────
models = {
    'Linear Regression':     LinearRegression(),
    'Ridge Regression':      Ridge(alpha=1.0),
    'Random Forest':         RandomForestRegressor(n_estimators=200, max_depth=8,
                                                    min_samples_leaf=10, random_state=42),
    'Gradient Boosting':     GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                                        learning_rate=0.1, random_state=42),
}

results = {}
for name, model in models.items():
    X_tr = X_train_s if 'Regression' in name else X_train.values
    X_te = X_test_s  if 'Regression' in name else X_test.values
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    X_cv = X_train_s if 'Regression' in name else X_train.values
    cv_r2 = cross_val_score(model, X_cv, y_train, cv=5, scoring='r2').mean()
    results[name] = {'MAE': round(mae,4), 'RMSE': round(rmse,4),
                     'R2': round(r2,4), 'CV_R2': round(cv_r2,4),
                     'y_pred': y_pred}
    print(f"{name}: MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f} | CV-R²={cv_r2:.4f}")

BEST = 'Gradient Boosting'
best_model = models[BEST]
best_res   = results[BEST]
y_pred_best = best_res['y_pred']

# Back-transform: log_spend → ₹
y_test_inr  = np.expm1(y_test.values)
y_pred_inr  = np.expm1(y_pred_best)
mae_inr  = mean_absolute_error(y_test_inr, y_pred_inr)
rmse_inr = np.sqrt(mean_squared_error(y_test_inr, y_pred_inr))
print(f"\nBack-transformed ({BEST}): MAE=₹{mae_inr:,.0f} | RMSE=₹{rmse_inr:,.0f}")

# ── Plot 1: Predicted vs Actual (log scale) ───────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test.values, y_pred_best, alpha=0.3, s=12, color='#7F77DD', edgecolors='none')
mn, mx = min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())
ax.plot([mn, mx], [mn, mx], '--', color='#E24B4A', linewidth=1.5, label='Perfect fit')
ax.set_xlabel('Actual log spend')
ax.set_ylabel('Predicted log spend')
ax.set_title(f'Predicted vs Actual — {BEST} (R²={best_res["R2"]:.3f})')
ax.legend()
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'regression_pred_actual.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Plot 2: Residuals ────────────────────────────────────────────
residuals = y_test.values - y_pred_best
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(y_pred_best, residuals, alpha=0.3, s=12,
                color='#1D9E75', edgecolors='none')
axes[0].axhline(0, color='#E24B4A', linestyle='--')
axes[0].set_xlabel('Predicted log spend')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted')
axes[0].spines[['top','right']].set_visible(False)
axes[1].hist(residuals, bins=40, color='#EF9F27', edgecolor='white', linewidth=0.3)
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Residual distribution')
axes[1].spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'regression_residuals.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Plot 3: Feature importance ────────────────────────────────────
fi = pd.Series(best_model.feature_importances_, index=REG_FEATURES).sort_values(ascending=True)
top12 = fi.tail(12)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top12.index, top12.values, color='#1D9E75')
ax.set_xlabel('Feature importance')
ax.set_title(f'Top 12 features — spend prediction ({BEST})')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'regression_importance.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Plot 4: Model comparison ─────────────────────────────────────
comp_df = pd.DataFrame({n: {'MAE': r['MAE'], 'RMSE': r['RMSE'], 'R²': r['R2']}
                         for n, r in results.items()}).T
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, (col, color) in enumerate(zip(['MAE','RMSE','R²'],['#D85A30','#E24B4A','#1D9E75'])):
    bars = axes[i].bar(comp_df.index, comp_df[col], color=color)
    axes[i].set_title(col)
    axes[i].set_xticklabels(comp_df.index, rotation=20, ha='right', fontsize=9)
    axes[i].spines[['top','right']].set_visible(False)
    for bar, v in zip(bars, comp_df[col]):
        axes[i].text(bar.get_x()+bar.get_width()/2, v+0.002,
                     f'{v:.3f}', ha='center', fontsize=8)
plt.suptitle('Regression model comparison', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'regression_comparison.png'), dpi=120, bbox_inches='tight')
plt.close()

# ── Save ─────────────────────────────────────────────────────────
with open(os.path.join(MODELS, 'regressor.pkl'), 'wb') as f:
    pickle.dump({'model': best_model, 'features': REG_FEATURES,
                 'scaler': scaler, 'name': BEST}, f)

save_metrics = {n: {k: v for k, v in r.items() if k != 'y_pred'}
                for n, r in results.items()}
save_metrics['back_transform'] = {'MAE_INR': round(float(mae_inr),2),
                                   'RMSE_INR': round(float(rmse_inr),2)}
with open(os.path.join(MODELS, 'regression_metrics.json'), 'w') as f:
    json.dump(save_metrics, f, indent=2)

print(f"\n✓ Regression complete. Best: {BEST} | R²={best_res['R2']:.4f}")
