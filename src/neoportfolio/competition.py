"""
NeoPortfolio — Competition matrix.
Quantifies pairwise immunodominance competition between neoantigen epitopes.

Biology:
  - Peptides restricted to the SAME HLA compete for the same MHC molecules.
  - Peptides with SIMILAR presentation strength produce unpredictable hierarchy.

Competition(i,j) = HLA_shared(i,j) x (0.6 + 0.4 x strength_similarity(i,j))

Range:
  1.0 = same HLA + same strength (worst case for co-delivery)
  0.6 = same HLA + very different strength (one dominates, other suppressed)
  0.0 = different HLA (no direct competition)
"""
import numpy as np
import pandas as pd


def _hla_shared(h1, h2):
    s1 = set(str(h1).replace(",", " ").split())
    s2 = set(str(h2).replace(",", " ").split())
    if not s1 or not s2:
        return 0.0
    return 1.0 if (s1 & s2) else 0.0


def build_competition_matrix(df, peptide_col="peptide", hla_col="hla",
                              strength_col="presentation_score"):
    peptides = df[peptide_col].tolist()
    n = len(peptides)
    hla_arr = df[hla_col].astype(str).tolist()
    strength = df[strength_col].astype(float).fillna(0).values

    s_min, s_max = strength.min(), strength.max()
    s_range = (s_max - s_min) + 1e-9
    s_norm = (strength - s_min) / s_range

    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
                continue
            hla_s = _hla_shared(hla_arr[i], hla_arr[j])
            if hla_s == 0.0:
                mat[i, j] = 0.0
            else:
                strength_sim = 1.0 - abs(s_norm[i] - s_norm[j])
                mat[i, j] = round(0.6 + 0.4 * strength_sim, 4)

    return pd.DataFrame(mat, index=peptides, columns=peptides)


def competition_summary(df, matrix):
    n = len(matrix)
    if n < 2:
        return {"n_peptides": n, "avg_competition": 0.0,
                "max_competition": 0.0, "hla_groups": 0}
    vals = matrix.values[np.triu_indices(n, k=1)]
    hla_groups = df["hla"].nunique() if "hla" in df.columns else 0
    return {
        "n_peptides":       n,
        "avg_competition":  round(float(vals.mean()), 4),
        "max_competition":  round(float(vals.max()), 4),
        "min_competition":  round(float(vals.min()), 4),
        "hla_groups":       hla_groups,
    }
