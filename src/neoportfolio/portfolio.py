"""
NeoPortfolio — Vaccine portfolio optimizer.

Exhaustive-search optimizer. Given N candidate epitopes, select the
K-subset that maximizes protective coverage while minimizing internal
immunodominance competition.

Objective:
  score(S) = sum(presentation_i for i in S)
           - lambda * sum(competition(i,j) for i<j in S)

With K = 6 (typical personalized vaccine size) and N = 13, we enumerate
all C(13,6) = 1716 subsets — trivial compute, globally optimal.
"""
from itertools import combinations
import pandas as pd
import numpy as np


def _portfolio_score(subset_idx, presentation, comp_matrix, lam=1.0):
    """
    Score a subset: total presentation - lam * total pairwise competition.
    Higher is better.
    """
    pres = presentation[list(subset_idx)].sum()
    if len(subset_idx) < 2:
        return float(pres)
    sub_matrix = comp_matrix[np.ix_(subset_idx, subset_idx)]
    n = len(subset_idx)
    comp_sum = sub_matrix[np.triu_indices(n, k=1)].sum()
    return float(pres - lam * comp_sum)


def optimize_portfolio(stab_df, comp_matrix, k=6, lam=1.0,
                        min_presentation=0.0, verbose=True):
    """
    Exhaustive search over all C(N, k) subsets.

    min_presentation: filter out peptides below this presentation score
                      before optimization (prevents selecting non-presented
                      peptides just for HLA diversity).
    """
    peptides = stab_df["peptide"].tolist()
    hla = stab_df["hla"].tolist()
    presentation = stab_df["presentation_score"].fillna(0).values
    comp_values = comp_matrix.values
    n = len(peptides)

    # Apply presentation threshold
    eligible_idx = [i for i in range(n) if presentation[i] >= min_presentation]
    if len(eligible_idx) < k:
        raise ValueError(
            f"Only {len(eligible_idx)} peptides pass threshold "
            f"{min_presentation}; need at least {k}. Lower the threshold."
        )

    n_eligible = len(eligible_idx)
    if verbose and min_presentation > 0:
        print(f"  Threshold {min_presentation}: {n_eligible}/{n} peptides eligible")

    best_score = -np.inf
    best_subset = None
    n_subsets = 0

    for local_subset in combinations(range(n_eligible), k):
        # Map local indices to global indices
        subset = tuple(eligible_idx[i] for i in local_subset)
        score = _portfolio_score(subset, presentation, comp_values, lam)
        n_subsets += 1
        if score > best_score:
            best_score = score
            best_subset = subset

    # Extract winners
    selected = [peptides[i] for i in best_subset]
    selected_idx = list(best_subset)

    # Diagnostics
    total_pres = float(presentation[selected_idx].sum())
    if k > 1:
        sub_matrix = comp_values[np.ix_(selected_idx, selected_idx)]
        pairwise_comp = sub_matrix[np.triu_indices(k, k=1)]
        avg_comp = float(pairwise_comp.mean())
        max_comp = float(pairwise_comp.max())
    else:
        avg_comp = 0.0
        max_comp = 0.0

    if verbose:
        print(f"  Enumerated {n_subsets:,} subsets")
        print(f"  Best score: {best_score:.4f}")
        print(f"  Total presentation: {total_pres:.4f}")
        print(f"  Avg pairwise competition: {avg_comp:.4f}")
        print(f"  Max pairwise competition: {max_comp:.4f}")

    return {
        "selected_peptides":     selected,
        "selected_indices":      selected_idx,
        "selected_hla":          [hla[i] for i in selected_idx],
        "selected_presentation": [float(presentation[i]) for i in selected_idx],
        "objective":             float(best_score),
        "total_presentation":    total_pres,
        "avg_competition":       avg_comp,
        "max_competition":       max_comp,
        "k":                     k,
        "n_subsets_enumerated":  n_subsets,
    }


def baseline_top_k(stab_df, k=6):
    """Naive baseline: top K by presentation score alone (no portfolio logic)."""
    ranked = stab_df.sort_values("presentation_score", ascending=False).head(k)
    return {
        "selected_peptides":     ranked["peptide"].tolist(),
        "selected_hla":          ranked["hla"].tolist(),
        "selected_presentation": ranked["presentation_score"].tolist(),
        "total_presentation":    float(ranked["presentation_score"].sum()),
    }


def compare_portfolio_vs_baseline(stab_df, comp_matrix, k=6, lam=1.0):
    """Run both approaches and return a side-by-side comparison."""
    portfolio = optimize_portfolio(stab_df, comp_matrix, k=k, lam=lam, verbose=False)
    baseline = baseline_top_k(stab_df, k=k)

    # Compute competition for baseline
    peptides = stab_df["peptide"].tolist()
    baseline_idx = [peptides.index(p) for p in baseline["selected_peptides"]]
    sub = comp_matrix.values[np.ix_(baseline_idx, baseline_idx)]
    baseline_comp = sub[np.triu_indices(k, k=1)]

    return {
        "portfolio":           portfolio,
        "baseline":            baseline,
        "baseline_avg_comp":   float(baseline_comp.mean()),
        "baseline_max_comp":   float(baseline_comp.max()),
        "competition_reduction_pct": (1 - portfolio["avg_competition"] / (baseline_comp.mean() + 1e-9)) * 100,
    }
