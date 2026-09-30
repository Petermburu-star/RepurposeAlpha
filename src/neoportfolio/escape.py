"""
NeoPortfolio — Escape-resilience scoring.

Models peptide vulnerability to tumor antigen-processing escape mechanisms,
following the CRUK Cambridge Institute framework (JCO 2026, DOI 10.1200/jco.2026.44.16_suppl.2654).

Five mechanisms:
1. TAP downregulation
2. Immunoproteasome-to-constitutive proteasome switching
3. Aminopeptidase (ERAP1) upregulation
4. Tapasin loss
5. HLA loss of heterozygosity (population-level, not per-peptide)

Per-peptide components computed from:
- DeepTAP (TAP transport propensity) — via mhctools
- NetChop / Pepsickle (proteasome cleavage propensity)
- ERAMER (ERAP1 trimming propensity)
- SPEARMINT (stability)

Higher escape_resilience_score = more robust to escape (better vaccine candidate).
"""
import numpy as np
import pandas as pd
from typing import Optional


# Predictor availability flags — populated on first use
_AVAILABLE = {}


def _try_load(predictor_name: str):
    """Attempt to load a predictor; return None if unavailable."""
    if predictor_name in _AVAILABLE:
        return _AVAILABLE[predictor_name]

    try:
        if predictor_name == "deeptap":
            from mhctools import DeepTAP
            DeepTAP.fetch()
            _AVAILABLE[predictor_name] = DeepTAP(task_type="cla")
        elif predictor_name == "pepsickle":
            from mhctools import Pepsickle
            _AVAILABLE[predictor_name] = Pepsickle()
        elif predictor_name == "eramer":
            from mhctools import ERAMER
            ERAMER.fetch()
            _AVAILABLE[predictor_name] = ERAMER()
        elif predictor_name == "netchop":
            from mhctools import NetChop
            _AVAILABLE[predictor_name] = NetChop()
        else:
            _AVAILABLE[predictor_name] = None
    except Exception as e:
        print(f"  ⚠️  {predictor_name} unavailable: {str(e)[:80]}")
        _AVAILABLE[predictor_name] = None

    return _AVAILABLE[predictor_name]


def compute_escape_components(peptides: list, c_flanks: Optional[list] = None) -> pd.DataFrame:
    """
    Compute per-peptide escape-resilience components.

    c_flanks: C-terminal flanking residues for each peptide (needed by cleavage/ERAP predictors).
              If None, uses a neutral flank "GGG".
    """
    n = len(peptides)
    peptides = [p.upper().strip() for p in peptides]

    if c_flanks is None:
        c_flanks = ["GGG"] * n

    df = pd.DataFrame({"peptide": peptides, "c_flank": c_flanks})

    # --- TAP transport (DeepTAP) ---
    tap = _try_load("deeptap")
    if tap is not None:
        try:
            res = tap.predict(peptides)
            df["tap_score"] = [r.tap_transport.score for r in res]
        except Exception as e:
            print(f"  ⚠️  DeepTAP prediction failed: {e}")
            df["tap_score"] = np.nan
    else:
        df["tap_score"] = np.nan

    # --- Proteasome cleavage (Pepsickle) ---
    pepsickle = _try_load("pepsickle")
    if pepsickle is not None:
        try:
            res = pepsickle.predict(peptides)
            # Pepsickle returns 'cleavage' kind; higher = more likely cleaved
            df["cleavage_score"] = [r.proteasome_cleavage.score for r in res]
        except Exception as e:
            print(f"  ⚠️  Pepsickle prediction failed: {e}")
            df["cleavage_score"] = np.nan
    else:
        df["cleavage_score"] = np.nan

    # --- ERAP1 trimming (ERAMER) ---
    eramer = _try_load("eramer")
    if eramer is not None:
        try:
            res = eramer.predict(peptides)
            df["erap_score"] = [r.erap_trimming.score for r in res]
        except Exception as e:
            print(f"  ⚠️  ERAMER prediction failed: {e}")
            df["erap_score"] = np.nan
    else:
        df["erap_score"] = np.nan

    return df


def compute_escape_resilience(df_with_stability: pd.DataFrame,
                                peptides: list,
                                c_flanks: Optional[list] = None) -> pd.DataFrame:
    """
    Composite escape-resilience score.

    Components:
    - stability_score (from SPEARMINT)
    - tap_score (from DeepTAP) — higher = better TAP binding
    - cleavage_score (from Pepsickle) — higher = more efficient cleavage
    - erap_score (from ERAMER) — higher = less trimming vulnerability

    Composite: weighted average of z-scored components.
    """
    df = df_with_stability.copy().reset_index(drop=True)

    components_df = compute_escape_components(peptides, c_flanks)
    df = df.merge(components_df[["peptide", "tap_score", "cleavage_score", "erap_score"]],
                  on="peptide", how="left")

    # Weights from CRUK paper (TAP: 0.42 OR → highest, proteasome: 0.54 OR)
    # Normalized inverse-odds weights:
    weights = {"stability_score": 0.25, "tap_score": 0.30,
               "cleavage_score": 0.25, "erap_score": 0.20}

    # Z-score each component, handling all-NaN columns
    z_cols = []
    for col, w in weights.items():
        if col in df.columns and df[col].notna().sum() > 1:
            vals = df[col].astype(float)
            z = (vals - vals.mean()) / (vals.std() + 1e-9)
            df[f"z_{col}"] = z
            z_cols.append((f"z_{col}", w))

    # Composite
    if z_cols:
        composite = sum(df[z] * w for z, w in z_cols)
        total_w = sum(w for _, w in z_cols)
        df["escape_resilience_score"] = (composite / total_w).round(4)
    else:
        df["escape_resilience_score"] = np.nan

    return df
