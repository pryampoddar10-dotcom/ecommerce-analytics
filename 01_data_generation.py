import numpy as np
import pandas as pd
from scipy.special import expit

np.random.seed(42)
N = 2000

PERSONAS = ['urban_professional', 'budget_student', 'value_seeker', 'homemaker', 'premium_loyalist']
PERSONA_WEIGHTS = [0.30, 0.22, 0.20, 0.16, 0.12]
PERSONA_COUNTS = [600, 440, 400, 320, 240]

personas = np.repeat(PERSONAS, PERSONA_COUNTS)
np.random.shuffle(personas)
df = pd.DataFrame({'respondent_id': range(1001, 1001+N), 'persona': personas})

AGE_GROUPS = ['18-24', '25-34', '35-44', '45-54', '55+']
AGE_PROBS = {
    'urban_professional': [0.10, 0.60, 0.22, 0.06, 0.02],
    'budget_student':     [0.80, 0.16, 0.03, 0.01, 0.00],
    'value_seeker':       [0.08, 0.38, 0.34, 0.14, 0.06],
    'homemaker':          [0.05, 0.30, 0.40, 0.18, 0.07],
    'premium_loyalist':   [0.02, 0.18, 0.42, 0.28, 0.10],
}

GENDER_PROBS = {
    'urban_professional': [0.55, 0.42, 0.02, 0.01],
    'budget_student':     [0.50, 0.46, 0.03, 0.01],
    'value_seeker':       [0.58, 0.40, 0.01, 0.01],
    'homemaker':          [0.15, 0.82, 0.02, 0.01],
    'premium_loyalist':   [0.60, 0.37, 0.02, 0.01],
}
GENDERS = ['Male', 'Female', 'Non-binary', 'Prefer not to say']

CITY_TIERS = ['Metro (Tier 1)', 'Tier 2 city', 'Tier 3 city', 'Rural / town']
CITY_PROBS = {
    'urban_professional': [0.70, 0.22, 0.06, 0.02],
    'budget_student':     [0.45, 0.35, 0.15, 0.05],
    'value_seeker':       [0.15, 0.38, 0.32, 0.15],
    'homemaker':          [0.25, 0.35, 0.28, 0.12],
    'premium_loyalist':   [0.80, 0.15, 0.04, 0.01],
}

EDU_LEVELS = ['High school or below', 'Undergraduate', 'Postgraduate', 'Doctoral']
EDU_PROBS = {
    'urban_professional': [0.02, 0.40, 0.50, 0.08],
    'budget_student':     [0.10, 0.82, 0.08, 0.00],
    'value_seeker':       [0.20, 0.50, 0.26, 0.04],
    'homemaker':          [0.25, 0.50, 0.23, 0.02],
    'premium_loyalist':   [0.01, 0.28, 0.55, 0.16],
}

EMP_STATUS = ['Student', 'Salaried (private)', 'Salaried (govt.)', 'Self-employed', 'Homemaker', 'Unemployed']
EMP_PROBS = {
    'urban_professional': [0.00, 0.78, 0.14, 0.06, 0.01, 0.01],
    'budget_student':     [0.88, 0.06, 0.01, 0.03, 0.00, 0.02],
    'value_seeker':       [0.02, 0.30, 0.10, 0.48, 0.06, 0.04],
    'homemaker':          [0.00, 0.08, 0.05, 0.12, 0.72, 0.03],
    'premium_loyalist':   [0.00, 0.58, 0.15, 0.24, 0.02, 0.01],
}

FREQ_LABELS = ['Rarely', 'Occasionally', 'Monthly', 'Weekly', 'Daily or almost daily']
FREQ_PROBS = {
    'urban_professional': [0.02, 0.08, 0.25, 0.48, 0.17],
    'budget_student':     [0.05, 0.20, 0.45, 0.25, 0.05],
    'value_seeker':       [0.10, 0.35, 0.38, 0.14, 0.03],
    'homemaker':          [0.05, 0.28, 0.45, 0.18, 0.04],
    'premium_loyalist':   [0.01, 0.05, 0.20, 0.45, 0.29],
}

PLATFORMS = ['Amazon', 'Flipkart', 'Myntra', 'Meesho', 'Nykaa', 'Ajio', 'Brand website', 'Other']
PLATFORM_PROBS = {
    'urban_professional': [0.32, 0.25, 0.12, 0.05, 0.08, 0.08, 0.07, 0.03],
    'budget_student':     [0.25, 0.28, 0.14, 0.15, 0.06, 0.05, 0.04, 0.03],
    'value_seeker':       [0.20, 0.30, 0.10, 0.22, 0.05, 0.04, 0.05, 0.04],
    'homemaker':          [0.22, 0.27, 0.15, 0.18, 0.08, 0.04, 0.04, 0.02],
    'premium_loyalist':   [0.35, 0.20, 0.14, 0.03, 0.10, 0.10, 0.06, 0.02],
}

DEVICES = ['Smartphone', 'Laptop / desktop', 'Tablet']
DEVICE_PROBS = {
    'urban_professional': [0.60, 0.35, 0.05],
    'budget_student':     [0.82, 0.16, 0.02],
    'value_seeker':       [0.85, 0.12, 0.03],
    'homemaker':          [0.80, 0.16, 0.04],
    'premium_loyalist':   [0.52, 0.42, 0.06],
}

BROWSE_TIMES = ['Under 10 minutes', '10-30 minutes', '30-60 minutes', 'More than 1 hour']
BROWSE_PROBS = {
    'urban_professional': [0.30, 0.45, 0.18, 0.07],
    'budget_student':     [0.12, 0.35, 0.35, 0.18],
    'value_seeker':       [0.08, 0.28, 0.40, 0.24],
    'homemaker':          [0.15, 0.35, 0.35, 0.15],
    'premium_loyalist':   [0.38, 0.42, 0.15, 0.05],
}

ABANDON_FREQ = ['Never', 'Rarely', 'Sometimes', 'Often', 'Almost always']
ABANDON_PROBS = {
    'urban_professional': [0.20, 0.40, 0.28, 0.09, 0.03],
    'budget_student':     [0.05, 0.18, 0.35, 0.28, 0.14],
    'value_seeker':       [0.06, 0.20, 0.35, 0.26, 0.13],
    'homemaker':          [0.10, 0.30, 0.38, 0.16, 0.06],
    'premium_loyalist':   [0.28, 0.45, 0.20, 0.05, 0.02],
}

ABANDON_REASONS = ['High price / no discount', 'High delivery charges', 'Just browsing',
                   'Found better deal elsewhere', 'Payment issues', 'N/A - I rarely abandon']
ABANDON_REASON_PROBS = {
    'urban_professional': [0.15, 0.20, 0.25, 0.15, 0.05, 0.20],
    'budget_student':     [0.38, 0.22, 0.20, 0.12, 0.05, 0.03],
    'value_seeker':       [0.35, 0.25, 0.18, 0.15, 0.04, 0.03],
    'homemaker':          [0.30, 0.28, 0.20, 0.12, 0.05, 0.05],
    'premium_loyalist':   [0.10, 0.12, 0.22, 0.10, 0.04, 0.42],
}

# Category co-occurrence probs per persona
CAT_PROBS = {
    # fashion, electronics, grocery, beauty, home, books, sports, toys
    'urban_professional': [0.55, 0.72, 0.40, 0.38, 0.45, 0.50, 0.60, 0.15],
    'budget_student':     [0.50, 0.65, 0.55, 0.30, 0.20, 0.62, 0.40, 0.10],
    'value_seeker':       [0.48, 0.55, 0.65, 0.32, 0.42, 0.30, 0.28, 0.22],
    'homemaker':          [0.68, 0.35, 0.78, 0.65, 0.70, 0.28, 0.22, 0.55],
    'premium_loyalist':   [0.60, 0.78, 0.48, 0.50, 0.65, 0.55, 0.68, 0.20],
}

DISCOVERY = ['Platform recommendations', 'Social media ads', 'Friends / family',
             'Search engine', 'Email / app notifications', 'Influencer reviews']
DISCOVERY_PROBS = {
    'urban_professional': [0.30, 0.18, 0.12, 0.28, 0.08, 0.04],
    'budget_student':     [0.18, 0.28, 0.18, 0.15, 0.06, 0.15],
    'value_seeker':       [0.20, 0.22, 0.20, 0.22, 0.10, 0.06],
    'homemaker':          [0.15, 0.22, 0.32, 0.14, 0.10, 0.07],
    'premium_loyalist':   [0.32, 0.15, 0.10, 0.30, 0.10, 0.03],
}

INCOME_BRACKETS = ['Below 20000', '20000-50000', '50000-100000', '100000-200000', 'Above 200000']
INCOME_PROBS = {
    'urban_professional': [0.02, 0.10, 0.38, 0.38, 0.12],
    'budget_student':     [0.30, 0.45, 0.20, 0.04, 0.01],
    'value_seeker':       [0.08, 0.38, 0.38, 0.13, 0.03],
    'homemaker':          [0.10, 0.40, 0.36, 0.11, 0.03],
    'premium_loyalist':   [0.00, 0.03, 0.18, 0.48, 0.31],
}
INCOME_NUM = {'Below 20000': 1, '20000-50000': 2, '50000-100000': 3, '100000-200000': 4, 'Above 200000': 5}

SPEND_BRACKETS = ['Under 500', '500-2000', '2000-5000', '5000-15000', 'Above 15000']
SPEND_NUM = {'Under 500': 1, '500-2000': 2, '2000-5000': 3, '5000-15000': 4, 'Above 15000': 5}

PAYMENT_METHODS = ['UPI', 'Debit card', 'Credit card', 'Net banking', 'Cash on delivery', 'Buy now pay later']
PAYMENT_PROBS = {
    'urban_professional': [0.42, 0.18, 0.25, 0.08, 0.04, 0.03],
    'budget_student':     [0.48, 0.22, 0.08, 0.05, 0.14, 0.03],
    'value_seeker':       [0.35, 0.18, 0.10, 0.06, 0.28, 0.03],
    'homemaker':          [0.38, 0.20, 0.12, 0.05, 0.22, 0.03],
    'premium_loyalist':   [0.30, 0.15, 0.40, 0.10, 0.02, 0.03],
}

LOYALTY_OPTIONS = ['Yes - one programme', 'Yes - multiple programmes', 'No but considering', 'No not interested']
LOYALTY_PROBS = {
    'urban_professional': [0.45, 0.25, 0.22, 0.08],
    'budget_student':     [0.12, 0.04, 0.38, 0.46],
    'value_seeker':       [0.18, 0.06, 0.30, 0.46],
    'homemaker':          [0.22, 0.08, 0.30, 0.40],
    'premium_loyalist':   [0.30, 0.58, 0.10, 0.02],
}
LOYALTY_MEMBER_FLAG = {'Yes - one programme': 1, 'Yes - multiple programmes': 1,
                        'No but considering': 0, 'No not interested': 0}

RETURN_FREQ = ['Never', 'Rarely', 'Occasionally', 'Frequently']
RETURN_PROBS = {
    'urban_professional': [0.15, 0.45, 0.32, 0.08],
    'budget_student':     [0.18, 0.42, 0.30, 0.10],
    'value_seeker':       [0.20, 0.42, 0.28, 0.10],
    'homemaker':          [0.12, 0.38, 0.36, 0.14],
    'premium_loyalist':   [0.20, 0.50, 0.25, 0.05],
}

SOCIAL_INFLUENCE = ['Never', 'Rarely', 'Sometimes', 'Often', 'Almost always']
SOCIAL_INF_PROBS = {
    'urban_professional': [0.08, 0.28, 0.38, 0.20, 0.06],
    'budget_student':     [0.04, 0.15, 0.35, 0.32, 0.14],
    'value_seeker':       [0.06, 0.20, 0.38, 0.26, 0.10],
    'homemaker':          [0.04, 0.18, 0.36, 0.30, 0.12],
    'premium_loyalist':   [0.12, 0.32, 0.36, 0.15, 0.05],
}

SOCIAL_COMMERCE = ['Yes regularly', 'Yes occasionally', 'No but open to it', 'No not interested']
SOCIAL_COM_PROBS = {
    'urban_professional': [0.20, 0.40, 0.28, 0.12],
    'budget_student':     [0.28, 0.38, 0.25, 0.09],
    'value_seeker':       [0.15, 0.32, 0.30, 0.23],
    'homemaker':          [0.18, 0.35, 0.30, 0.17],
    'premium_loyalist':   [0.15, 0.38, 0.32, 0.15],
}

FESTIVE_SPEND = ['Less than 10pct', '10-25pct', '26-50pct', 'More than 50pct']
FESTIVE_PROBS = {
    'urban_professional': [0.18, 0.38, 0.32, 0.12],
    'budget_student':     [0.20, 0.35, 0.30, 0.15],
    'value_seeker':       [0.08, 0.22, 0.40, 0.30],
    'homemaker':          [0.10, 0.28, 0.38, 0.24],
    'premium_loyalist':   [0.25, 0.40, 0.28, 0.07],
}

IMPULSE_TRIGGERS = ['Festive sale', 'Flash sale notification', 'Influencer post',
                    'Boredom', 'Gifting occasion', 'Recommendation email']
IMPULSE_PROBS = {
    'urban_professional': [0.20, 0.25, 0.10, 0.10, 0.22, 0.13],
    'budget_student':     [0.22, 0.28, 0.20, 0.15, 0.10, 0.05],
    'value_seeker':       [0.28, 0.30, 0.12, 0.10, 0.14, 0.06],
    'homemaker':          [0.30, 0.22, 0.15, 0.08, 0.20, 0.05],
    'premium_loyalist':   [0.18, 0.20, 0.08, 0.06, 0.28, 0.20],
}

PERSONA_INTERCEPTS = {
    'urban_professional': 0.8,
    'budget_student':    -0.3,
    'value_seeker':      -0.5,
    'homemaker':         -0.1,
    'premium_loyalist':   1.2,
}

def sample_col(persona, probs_dict, choices):
    p = probs_dict[persona]
    return np.random.choice(choices, p=p)

def ordinal_score(value, scale):
    return scale.index(value) + 1 if value in scale else 3

rows = []
for i, p in enumerate(df['persona']):
    r = {}
    r['respondent_id'] = df['respondent_id'].iloc[i]
    r['persona'] = p

    r['age_group']         = sample_col(p, AGE_PROBS, AGE_GROUPS)
    r['gender']            = sample_col(p, GENDER_PROBS, GENDERS)
    r['city_tier']         = sample_col(p, CITY_PROBS, CITY_TIERS)
    r['education_level']   = sample_col(p, EDU_PROBS, EDU_LEVELS)
    r['employment_status'] = sample_col(p, EMP_PROBS, EMP_STATUS)

    r['shopping_frequency']         = sample_col(p, FREQ_PROBS, FREQ_LABELS)
    plat_probs = PLATFORM_PROBS[p]
    plat_choices = np.random.choice(PLATFORMS, size=3, replace=False, p=plat_probs)
    r['platform_1'] = plat_choices[0]
    r['platform_2'] = plat_choices[1]
    r['platform_3'] = plat_choices[2] if np.random.rand() > 0.30 else np.nan
    r['primary_device']             = sample_col(p, DEVICE_PROBS, DEVICES)
    r['browse_time_before_purchase']= sample_col(p, BROWSE_PROBS, BROWSE_TIMES)
    r['cart_abandonment_freq']      = sample_col(p, ABANDON_PROBS, ABANDON_FREQ)
    r['cart_abandon_reason']        = sample_col(p, ABANDON_REASON_PROBS, ABANDON_REASONS)

    cat_p = CAT_PROBS[p]
    cats = ['cat_fashion','cat_electronics','cat_grocery','cat_beauty',
            'cat_home','cat_books','cat_sports','cat_toys']
    for j, cat in enumerate(cats):
        r[cat] = int(np.random.rand() < cat_p[j])

    r['review_importance_score'] = np.random.choice([1,2,3,4,5],
        p=[0.03,0.07,0.20,0.38,0.32] if p in ['value_seeker','homemaker'] else
          [0.02,0.05,0.18,0.40,0.35] if p=='budget_student' else
          [0.02,0.08,0.22,0.40,0.28])
    r['discovery_channel']              = sample_col(p, DISCOVERY_PROBS, DISCOVERY)
    r['recommendation_acceptance_score']= np.random.choice([1,2,3,4,5],
        p=[0.02,0.05,0.15,0.40,0.38] if p in ['urban_professional','premium_loyalist'] else
          [0.05,0.12,0.30,0.32,0.21] if p=='budget_student' else
          [0.08,0.18,0.32,0.28,0.14])

    income_label = sample_col(p, INCOME_PROBS, INCOME_BRACKETS)
    r['monthly_income_bracket'] = income_label
    income_num = INCOME_NUM[income_label]

    spend_noise = np.random.randint(-1, 2)
    spend_num = max(1, min(5, income_num + spend_noise))
    r['monthly_spend_bracket'] = SPEND_BRACKETS[spend_num - 1]

    MU_MAP = {1: 6.5, 2: 7.5, 3: 8.4, 4: 9.4, 5: 10.5}
    mu = MU_MAP[income_num]
    r['max_single_transaction_value'] = round(np.random.lognormal(mu, 0.55), 2)

    disc_base = {1:4.5, 2:4.0, 3:3.2, 4:2.4, 5:1.6}[income_num]
    disc_score = int(np.clip(round(disc_base + np.random.normal(0, 0.7)), 1, 5))
    r['discount_sensitivity_score'] = disc_score

    imp_thresh_probs = {
        1:[0.05,0.10,0.25,0.35,0.25],
        2:[0.08,0.15,0.30,0.30,0.17],
        3:[0.12,0.22,0.32,0.24,0.10],
        4:[0.22,0.30,0.28,0.14,0.06],
        5:[0.38,0.32,0.20,0.08,0.02],
    }
    THRESH_LABELS = ['Discounts dont trigger', '10-20pct', '21-40pct', '41-60pct', 'Above 60pct']
    r['impulse_discount_threshold'] = np.random.choice(THRESH_LABELS, p=imp_thresh_probs[disc_score])

    r['preferred_payment_method'] = sample_col(p, PAYMENT_PROBS, PAYMENT_METHODS)
    loyalty_label = sample_col(p, LOYALTY_PROBS, LOYALTY_OPTIONS)
    r['loyalty_member']           = loyalty_label
    r['loyalty_member_flag']      = LOYALTY_MEMBER_FLAG[loyalty_label]

    r['delivery_importance_score'] = np.random.choice([1,2,3,4,5], p=[0.02,0.05,0.15,0.38,0.40])
    r['return_frequency']          = sample_col(p, RETURN_PROBS, RETURN_FREQ)

    r['social_purchase_influence_freq'] = sample_col(p, SOCIAL_INF_PROBS, SOCIAL_INFLUENCE)
    r['social_commerce_adoption']       = sample_col(p, SOCIAL_COM_PROBS, SOCIAL_COMMERCE)

    rpi_base = {'urban_professional':4.2,'budget_student':2.8,'value_seeker':2.5,'homemaker':3.0,'premium_loyalist':4.6}[p]
    r['repeat_purchase_intent_score'] = int(np.clip(round(rpi_base + np.random.normal(0, 0.7)), 1, 5))

    loyalty_flag = r['loyalty_member_flag']
    swp_base = 3.5 - loyalty_flag * 1.5
    r['platform_switching_propensity'] = int(np.clip(round(swp_base + np.random.normal(0, 0.8)), 1, 5))

    r['review_writing_freq']   = sample_col(p, {
        'urban_professional': [0.08,0.28,0.37,0.27],
        'budget_student':     [0.10,0.30,0.38,0.22],
        'value_seeker':       [0.12,0.32,0.38,0.18],
        'homemaker':          [0.10,0.28,0.40,0.22],
        'premium_loyalist':   [0.05,0.22,0.38,0.35],
    }, RETURN_FREQ)

    r['festive_spend_proportion'] = sample_col(p, FESTIVE_PROBS, FESTIVE_SPEND)
    r['impulse_trigger_type']     = sample_col(p, IMPULSE_PROBS, IMPULSE_TRIGGERS)

    ai_base = {'urban_professional':4.1,'budget_student':3.2,'value_seeker':2.4,'homemaker':2.6,'premium_loyalist':3.8}[p]
    r['ai_personalisation_comfort'] = int(np.clip(round(ai_base + np.random.normal(0, 0.8)), 1, 5))

    # TARGET: weighted sigmoid
    def norm5(x): return (x - 3) / 2.0
    score = (
        0.30 * norm5(r['repeat_purchase_intent_score']) +
        0.20 * norm5(r['recommendation_acceptance_score']) +
        0.15 * norm5(r['ai_personalisation_comfort']) -
        0.15 * norm5(r['platform_switching_propensity']) +
        0.10 * norm5(ordinal_score(r['shopping_frequency'], FREQ_LABELS)) -
        0.10 * norm5(ordinal_score(r['cart_abandonment_freq'], ABANDON_FREQ)) +
        0.05 * norm5(r['delivery_importance_score']) -
        0.05 * norm5(r['discount_sensitivity_score'])
    )
    intercept = PERSONA_INTERCEPTS[p]
    prob = expit(score * 2.5 + intercept)
    r['will_purchase'] = int(np.random.rand() < prob)

    rows.append(r)

df_out = pd.DataFrame(rows)

# ── Noise injection ──────────────────────────────────────────────
ORDINAL_COLS = ['review_importance_score','recommendation_acceptance_score',
                'discount_sensitivity_score','delivery_importance_score',
                'repeat_purchase_intent_score','platform_switching_propensity',
                'ai_personalisation_comfort']

noise_idx = np.random.choice(N, size=int(N * 0.02), replace=False)
for idx in noise_idx:
    cols_to_perturb = np.random.choice(ORDINAL_COLS, size=4, replace=False)
    for col in cols_to_perturb:
        df_out.at[idx, col] = np.random.randint(1, 6)

outlier_idx = np.random.choice(N, size=int(N * 0.01), replace=False)
for idx in outlier_idx:
    df_out.at[idx, 'max_single_transaction_value'] = round(
        df_out.at[idx, 'max_single_transaction_value'] * np.random.uniform(3, 8), 2)

flip_idx = np.random.choice(N, size=int(N * 0.01), replace=False)
for idx in flip_idx:
    df_out.at[idx, 'will_purchase'] = 1 - df_out.at[idx, 'will_purchase']

out_path = '/mnt/user-data/outputs/ecommerce_survey_synthetic.csv'
df_out.to_csv(out_path, index=False)

# ── Validation report ────────────────────────────────────────────
print(f"Shape: {df_out.shape}")
print(f"\nPersona distribution:\n{df_out['persona'].value_counts()}")
print(f"\nTarget balance:\n{df_out['will_purchase'].value_counts(normalize=True).round(3)}")
print(f"\nIncome brackets:\n{df_out['monthly_income_bracket'].value_counts()}")
print(f"\nMissing values:\n{df_out.isnull().sum()[df_out.isnull().sum()>0]}")
print(f"\nSpend stats:\n{df_out['max_single_transaction_value'].describe().round(2)}")
print(f"\nCity tier dist:\n{df_out['city_tier'].value_counts()}")
print(f"\nCategory purchase rates:")
for c in ['cat_fashion','cat_electronics','cat_grocery','cat_beauty','cat_home','cat_books','cat_sports','cat_toys']:
    print(f"  {c}: {df_out[c].mean():.3f}")
print(f"\nIncome-spend correlation: {df_out['monthly_income_bracket'].map(INCOME_NUM).corr(df_out['monthly_spend_bracket'].map(SPEND_NUM)):.3f}")
print(f"\nFile saved: {out_path}")
