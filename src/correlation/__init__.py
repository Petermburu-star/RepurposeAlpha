"""
RepurposeAlpha — generic correlation engine (with affinity threshold).

Only targets bound with IC50/Ki/EC50/Kd < 1 μM are considered meaningful.
This filters out safety-panel noise.
"""
import requests
import pandas as pd
import numpy as np
from typing import Optional

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

BINDING_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50"}
UNIT_TO_NM = {
    "nM": 1.0, "uM": 1000.0, "µM": 1000.0, "μM": 1000.0,
    "mM": 1_000_000.0, "pM": 0.001, "M": 1_000_000_000.0,
}

# The key fix: ignore weak binding (safety-panel noise)
AFFINITY_THRESHOLD_NM = 1000.0   # 1 μM

NOISE_TARGETS = {
    "Rattus norvegicus", "Myometrium", "Trypanosoma cruzi",
    "Trypanosoma brucei rhodesiense", "Trypanosoma brucei brucei",
    "Trypanosoma brucei", "Trichomonas vaginalis", "Toxoplasma gondii",
    "Pneumocystis carinii", "Plasmodium falciparum", "Leishmania",
}


def fetch_activities(chembl_id: str, timeout: int = 60) -> pd.DataFrame:
    url = f"{CHEMBL_BASE}/activity.json"
    params = {"molecule_chembl_id": chembl_id, "limit": 500}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return pd.DataFrame()
        rows = []
        for act in r.json().get("activities", []):
            rows.append({
                "chembl_id": chembl_id,
                "target_name": act.get("target_pref_name"),
                "standard_type": act.get("standard_type"),
                "standard_value": act.get("standard_value"),
                "standard_units": act.get("standard_units"),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def build_activity_table(chembl_ids: list, verbose: bool = True) -> pd.DataFrame:
    frames = []
    for i, cid in enumerate(chembl_ids, 1):
        if verbose:
            print(f"  [{i}/{len(chembl_ids)}] fetching {cid}...")
        df = fetch_activities(cid)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["standard_value"] = pd.to_numeric(combined["standard_value"], errors="coerce")
    combined = combined[combined["standard_type"].isin(BINDING_TYPES)].copy()
    combined = combined[~combined["target_name"].isin(NOISE_TARGETS)]
    combined = combined.dropna(subset=["standard_value", "target_name"])
    combined["value_nM"] = (
        combined["standard_value"]
        * combined["standard_units"].map(UNIT_TO_NM).fillna(1.0)
    )
    # THE FIX: filter to strong binders only
    combined = combined[combined["value_nM"] <= AFFINITY_THRESHOLD_NM]
    return combined


def build_correlation_matrix(chembl_ids: list, verbose: bool = True) -> pd.DataFrame:
    n = len(chembl_ids)
    if n < 2:
        return pd.DataFrame()

    if verbose:
        print(f"→ Building correlation matrix for {n} candidates...")

    activities = build_activity_table(chembl_ids, verbose=verbose)
    if activities.empty:
        if verbose:
            print("  ❌ No bioactivity data after filtering")
        return pd.DataFrame()

    targets = sorted(activities["target_name"].unique())
    binary = pd.DataFrame(0, index=chembl_ids, columns=targets)
    for _, r in activities.iterrows():
        binary.loc[r["chembl_id"], r["target_name"]] = 1

    median_aff = activities.groupby(
        ["chembl_id", "target_name"]
    )["value_nM"].median().to_dict()

    def jaccard(a, b):
        inter = int(((a == 1) & (b == 1)).sum())
        union = int(((a == 1) | (b == 1)).sum())
        return inter / union if union > 0 else 0.0

    def affinity_sim(d1, d2):
        if d1 == d2:
            return 1.0
        a, b = binary.loc[d1].values, binary.loc[d2].values
        shared = np.array(targets)[(a == 1) & (b == 1)]
        if shared.size == 0:
            return 0.0
        sims = []
        for t in shared:
            a1 = median_aff.get((d1, t))
            a2 = median_aff.get((d2, t))
            if pd.notna(a1) and pd.notna(a2) and a1 > 0 and a2 > 0:
                sims.append(np.exp(-abs(np.log10(a1) - np.log10(a2))))
        return float(np.mean(sims)) if sims else 0.0

    mat = pd.DataFrame(np.zeros((n, n)), index=chembl_ids, columns=chembl_ids)
    for d1 in chembl_ids:
        for d2 in chembl_ids:
            if d1 == d2:
                mat.loc[d1, d2] = 1.0
            else:
                j = jaccard(binary.loc[d1].values, binary.loc[d2].values)
                a = affinity_sim(d1, d2)
                mat.loc[d1, d2] = 0.5 * j + 0.5 * a

    if verbose:
        print(f"  ✓ Correlation matrix built ({n}×{n})")
        print(f"  ✓ Bioactivity records (after ≤{AFFINITY_THRESHOLD_NM:.0f} nM filter): {len(activities)}")
        print(f"  ✓ Unique targets: {len(targets)}")

    return mat
