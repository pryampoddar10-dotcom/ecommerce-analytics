"""Lightweight Apriori implementation — no external dependencies."""
from itertools import combinations
import pandas as pd
import numpy as np

def apriori_scratch(basket_df, min_support=0.10, max_len=3):
    n = len(basket_df)
    items = list(basket_df.columns)
    freq_itemsets = []

    # 1-itemsets
    for item in items:
        sup = basket_df[item].mean()
        if sup >= min_support:
            freq_itemsets.append({'itemsets': frozenset([item]), 'support': sup})

    # k-itemsets
    for k in range(2, max_len+1):
        prev = [fs['itemsets'] for fs in freq_itemsets if len(fs['itemsets']) == k-1]
        candidates = set()
        for a, b in combinations(prev, 2):
            union = a | b
            if len(union) == k:
                candidates.add(union)
        for cand in candidates:
            cols = list(cand)
            sup = basket_df[cols].all(axis=1).mean()
            if sup >= min_support:
                freq_itemsets.append({'itemsets': cand, 'support': sup})

    df_out = pd.DataFrame(freq_itemsets)
    df_out['length'] = df_out['itemsets'].apply(len)
    return df_out

def association_rules_scratch(freq_df, min_confidence=0.40):
    rules = []
    two_plus = freq_df[freq_df['length'] >= 2]
    sup_dict = {fs: sup for fs, sup in zip(freq_df['itemsets'], freq_df['support'])}

    for _, row in two_plus.iterrows():
        itemset = row['itemsets']
        sup_itemset = row['support']
        for size in range(1, len(itemset)):
            for ant in combinations(sorted(itemset), size):
                ant = frozenset(ant)
                con = itemset - ant
                if ant in sup_dict and sup_dict[ant] > 0:
                    conf = sup_itemset / sup_dict[ant]
                    if conf >= min_confidence:
                        sup_ant = sup_dict[ant]
                        sup_con = sup_dict.get(con, 0)
                        lift = conf / sup_con if sup_con > 0 else 0
                        rules.append({
                            'antecedents': ant, 'consequents': con,
                            'support': round(sup_itemset, 4),
                            'confidence': round(conf, 4),
                            'lift': round(lift, 4)
                        })
    return pd.DataFrame(rules).sort_values('lift', ascending=False) if rules else pd.DataFrame()
