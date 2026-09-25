"""
RepurposeAlpha — per-drug return estimator (v3).
Rescales mu to realistic annual-return range for defensible Sharpe.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import assumptions

import pandas as pd
import numpy as np


STAGE_TO_KEY = {
    "APPROVAL": "approval", "PHASE_4": "phase_4", "PHASE_3": "phase_3",
    "PHASE_2_3": "phase_2_3", "PHASE_2": "phase_2",
    "PHASE_1_2": "phase_1_2", "PHASE_1": "phase_1", "PRECLINICAL": "preclinical",
}
PHASE_KEY_TO_POS = {
    "approval": "nda_bla", "phase_4": "nda_bla", "phase_3": "phase_3",
    "phase_2_3": "phase_2", "phase_2": "phase_2",
    "phase_1_2": "phase_1", "phase_1": "phase_1", "preclinical": "phase_1",
}
COST_PROXY = {
    "approval": 0.00, "phase_4": 0.00, "phase_3": 1.00, "phase_2_3": 0.65,
    "phase_2": 0.35, "phase_1_2": 0.20, "phase_1": 0.10, "preclinical": 0.05,
}

# Target annual return range (realistic for biotech portfolios)
ANNUAL_RETURN_MIN = 0.05    # 5% — low-risk approved drug
ANNUAL_RETURN_MAX = 0.20    # 20% — high-return approved drug


def _stage_to_key(stage):
    if not stage:
        return "phase_2"
    s = str(stage).upper().strip().replace(" ", "_")
    return STAGE_TO_KEY.get(s, "phase_2")


def estimate_returns(candidates, corr, verbose=True):
    rows = []
    peer_penalty = assumptions.get_value("mu_adjustment.peer_penalty", -0.10)
    unique_bonus = assumptions.get_value("mu_adjustment.unique_bonus", 0.15)

    for _, row in candidates.iterrows():
        cid = row["chembl_id"]
        if cid not in corr.index:
            continue

        phase_key = _stage_to_key(row.get("stage_str") or row.get("max_phase"))
        pos_key = PHASE_KEY_TO_POS.get(phase_key, "phase_2")
        pos = assumptions.get_value(f"pos.{pos_key}", 0.31)
        rev = assumptions.get_value(f"revenue_proxy.{phase_key}", 0.40)
        cost = COST_PROXY.get(phase_key, 0.35)

        # Raw rNPV proxy
        base_mu = pos * rev - (1 - pos) * cost

        # Peer adjustment
        peers = int((corr.loc[cid] > 0.3).sum() - 1)
        peer_adj = unique_bonus if peers == 0 else peer_penalty * min(peers, 3)
        mu_raw = base_mu + peer_adj

        # Volatility
        base_vol = assumptions.get_value(f"volatility_base.{phase_key}", 0.30)
        if peers == 0:
            vol_adj = assumptions.get_value("novelty_adjustment.unique_mechanism", 0.05)
        elif peers >= 5:
            vol_adj = assumptions.get_value("novelty_adjustment.crowded_mechanism", -0.05)
        else:
            vol_adj = 0.0
        sigma = float(np.clip(base_vol + vol_adj, 0.05, 0.60))

        rows.append({
            "chembl_id": cid,
            "drug_name": row.get("drug_name", cid),
            "phase_key": phase_key,
            "pos": pos,
            "revenue_proxy": rev,
            "cost_proxy": cost,
            "peers": peers,
            "mu_raw": round(mu_raw, 4),
            "sigma": round(sigma, 3),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ---- RESCALE raw mu to realistic annual return range ----
    mu_min, mu_max = df["mu_raw"].min(), df["mu_raw"].max()
    if mu_max > mu_min:
        df["mu"] = ANNUAL_RETURN_MIN + (ANNUAL_RETURN_MAX - ANNUAL_RETURN_MIN) * \
                   (df["mu_raw"] - mu_min) / (mu_max - mu_min)
    else:
        df["mu"] = (ANNUAL_RETURN_MIN + ANNUAL_RETURN_MAX) / 2

    df["mu"] = df["mu"].round(4)
    df = df.set_index("chembl_id")

    if verbose:
        print(f"  Estimated returns for {len(df)} candidates")
        print(f"  mu (annual return) range: {df['mu'].min():.4f} – {df['mu'].max():.4f}")
        print(f"  sigma range:              {df['sigma'].min():.3f} – {df['sigma'].max():.3f}")
        print(f"  peers range:              {df['peers'].min()} – {df['peers'].max()}")

    return df
