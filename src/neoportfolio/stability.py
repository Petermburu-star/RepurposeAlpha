"""
NeoPortfolio — pMHC-I presentation score prediction.
Uses MHCflurry Class1PresentationPredictor, one call per pair.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple


_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from mhcflurry import Class1PresentationPredictor
    print("-> Loading MHCflurry Class1PresentationPredictor...")
    _MODEL = Class1PresentationPredictor.load()
    print("OK predictor loaded")
    return _MODEL


def _format_hla(hla):
    h = str(hla).upper().strip().replace("HLA-", "")
    if len(h) >= 2 and h[1] != "*":
        h = h[0] + "*" + h[1:]
    return h


def predict_stability(pairs):
    if not pairs:
        return pd.DataFrame()

    model = _load_model()
    rows = []

    for peptide, hla in pairs:
        p = peptide.upper().strip()
        a = _format_hla(hla)
        try:
            res = model.predict(peptides=[p], alleles=[a], verbose=0)
            rows.append({
                "peptide":            p,
                "hla":                a,
                "affinity_nM":        float(res["affinity"].iloc[0]) if "affinity" in res.columns else np.nan,
                "processing_score":   float(res["processing_score"].iloc[0]) if "processing_score" in res.columns else np.nan,
                "presentation_score": float(res["presentation_score"].iloc[0]) if "presentation_score" in res.columns else np.nan,
            })
        except Exception as e:
            print(f"  WARN {p} / {a}: {type(e).__name__}: {str(e)[:80]}")
            rows.append({
                "peptide": p, "hla": a,
                "affinity_nM": np.nan,
                "processing_score": np.nan,
                "presentation_score": np.nan,
            })

    out = pd.DataFrame(rows)
    out["stability_score"] = out["presentation_score"]
    out["stability_hours"] = (out["presentation_score"] * 20).clip(lower=0.1)
    return out
