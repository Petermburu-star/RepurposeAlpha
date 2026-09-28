"""
RepurposeAlpha — Combination synergy predictor.

Scores every pair of candidates for synergy potential:
1. Mechanism complementarity (different targets, shared context)
2. Prior combination trial evidence (ClinicalTrials.gov)
"""
import requests
import pandas as pd
import numpy as np
from itertools import combinations


CT_BASE = "https://clinicaltrials.gov/api/v2/studies"


def _complementarity_score(corr_value: float) -> float:
    """
    Synergy peaks when drugs share context but differ in mechanism.
    High correlation (>0.8) = redundant. Low (<0.05) = unrelated.
    Sweet spot = 0.1 - 0.5.
    """
    if corr_value < 0.05:
        return 0.30 + corr_value * 4.0
    if corr_value <= 0.50:
        return 0.80
    if corr_value <= 0.80:
        return max(0.20, 0.80 - (corr_value - 0.50) * 2.0)
    return 0.20


def _prior_combo_test(drug_a: str, drug_b: str) -> bool:
    """Check if a combination trial exists in ClinicalTrials.gov."""
    if not drug_a or not drug_b:
        return False
    query = f"{drug_a} AND {drug_b}"
    try:
        r = requests.get(
            CT_BASE,
            params={"query.intr": query, "pageSize": 5, "format": "json"},
            timeout=15,
        )
        if r.status_code == 200:
            return len(r.json().get("studies", [])) > 0
    except Exception:
        pass
    return False


def find_synergy_pairs(candidates_df, corr, top_n=10,
                        check_trials=False, verbose=True):
    """Score every pair for synergy potential."""
    if candidates_df is None or corr is None or corr.empty:
        return pd.DataFrame()

    drug_names = dict(zip(candidates_df["chembl_id"], candidates_df["drug_name"]))
    ids = [c for c in corr.columns if c in drug_names]

    pairs = []
    total = len(ids) * (len(ids) - 1) // 2
    idx = 0

    for a, b in combinations(ids, 2):
        idx += 1
        corr_val = float(corr.loc[a, b])
        comp = _complementarity_score(corr_val)

        prior = False
        if check_trials:
            prior = _prior_combo_test(drug_names[a], drug_names[b])

        synergy = 0.7 * comp + 0.3 * (1.0 if prior else 0.0)

        if prior:
            rationale = f"Prior combination trial exists (r={corr_val:.2f})"
        elif corr_val < 0.10:
            rationale = f"Unrelated mechanisms — complementary (r={corr_val:.2f})"
        elif corr_val <= 0.50:
            rationale = f"Complementary, shared context (r={corr_val:.2f})"
        elif corr_val <= 0.80:
            rationale = f"Overlapping mechanisms — possible redundancy (r={corr_val:.2f})"
        else:
            rationale = f"Redundant — same biological bet (r={corr_val:.2f})"

        pairs.append({
            "drug_a":          drug_names[a],
            "drug_b":          drug_names[b],
            "chembl_a":        a,
            "chembl_b":        b,
            "correlation":     round(corr_val, 3),
            "complementarity": round(comp, 3),
            "prior_combo":     prior,
            "synergy_score":   round(synergy, 3),
            "rationale":       rationale,
        })

    df = pd.DataFrame(pairs).sort_values("synergy_score", ascending=False).reset_index(drop=True)

    if verbose:
        n_prior = int(df["prior_combo"].sum())
        print(f"  Scored {len(df)} pairs | {n_prior} with prior combos")

    return df.head(top_n)


def top_combination_recommendation(synergy_df):
    if synergy_df is None or synergy_df.empty:
        return {"found": False}
    top = synergy_df.iloc[0]
    return {
        "found": True,
        "drug_a": top["drug_a"],
        "drug_b": top["drug_b"],
        "score": float(top["synergy_score"]),
        "rationale": top["rationale"],
    }
