"""
RepurposeAlpha — Combination synergy predictor (v2 with SYNERGxDB).

Real synergy scores from SYNERGxDB where available.
Falls back to heuristic (correlation-based) when the pair isn't in SYNERGxDB.

SYNERGxDB: https://www.synergxdb.ca — open access, no auth.
Coverage: cancer cell-line focused. Non-cancer pairs fall back to heuristic.
"""
import requests
import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
from functools import lru_cache
import time

SYNERGXDB_BASE = "https://www.synergxdb.ca/api"

# Cache location for the drug list (fetched once)
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DRUG_CACHE = CACHE_DIR / "synergxdb_drugs.parquet"


# ==================== Drug list management ====================

def _load_drug_list(verbose: bool = False) -> pd.DataFrame:
    """Load SYNERGxDB drug list, caching to disk."""
    if DRUG_CACHE.exists():
        if verbose:
            print(f"  ✓ Loaded cached SYNERGxDB drug list")
        return pd.read_parquet(DRUG_CACHE)

    if verbose:
        print(f"  → Fetching SYNERGxDB drug list...")
    r = requests.get(f"{SYNERGXDB_BASE}/drugs/", timeout=60)
    r.raise_for_status()
    drugs = pd.DataFrame(r.json())
    drugs.to_parquet(DRUG_CACHE, index=False)
    if verbose:
        print(f"  ✓ {len(drugs):,} drugs cached")
    return drugs


def _find_drug_id(drug_name: str, drug_df: pd.DataFrame) -> int:
    """
    Match a drug name to SYNERGxDB idDrug.
    Handles salt suffixes and case.
    """
    if not drug_name:
        return None

    # Normalize: uppercase, drop salt suffix words
    SALT_WORDS = {"HYDROCHLORIDE", "SODIUM", "POTASSIUM", "SULFATE", "CITRATE",
                  "TARTRATE", "PHOSPHATE", "MESYLATE", "BESYLATE", "MALEATE",
                  "FUMARATE", "ACETATE", "SUCCINATE", "TOSYLATE", "BROMIDE",
                  "CHLORIDE", "IODIDE", "CALCIUM", "MAGNESIUM", "MONOHYDRATE",
                  "DIHYDRATE", "ANHYDROUS", "HYDRATE"}

    words = str(drug_name).upper().split()
    base_name = " ".join(w for w in words if w not in SALT_WORDS).strip()

    if not base_name:
        return None

    names_upper = drug_df["name"].str.upper()

    # 1. Exact match
    exact = drug_df[names_upper == base_name]
    if not exact.empty:
        return int(exact.iloc[0]["idDrug"])

    # 2. Contains match (prefer shortest to avoid "Doxorubicin" matching "Doxorubicin Liposomal")
    contains = drug_df[names_upper.str.contains(base_name, regex=False, na=False)]
    if not contains.empty:
        # Sort by name length — shortest = likely the base drug
        contains = contains.assign(_len=contains["name"].str.len()).sort_values("_len")
        return int(contains.iloc[0]["idDrug"])

    return None


# ==================== Synergy queries ====================

@lru_cache(maxsize=512)
def _query_synergy(id_drug_1: int, id_drug_2: int) -> dict:
    """
    Query SYNERGxDB for a drug pair. Returns aggregated synergy stats.
    """
    try:
        r = requests.get(
            f"{SYNERGXDB_BASE}/combos/",
            params={"drugId1": id_drug_1, "drugId2": id_drug_2},
            timeout=30,
        )
        if r.status_code != 200:
            return {"found": False, "error": f"HTTP {r.status_code}"}
        records = r.json()
        if not records:
            return {"found": False, "n_studies": 0}

        df = pd.DataFrame(records)
        result = {"found": True, "n_studies": len(df)}

        # Aggregate each metric with median (robust to outliers)
        for metric in ["bliss", "loewe", "hsa", "zip", "comboscore"]:
            if metric in df.columns:
                vals = pd.to_numeric(df[metric], errors="coerce").dropna()
                if not vals.empty:
                    result[f"{metric}_median"] = float(vals.median())
                    result[f"{metric}_mean"] = float(vals.mean())

        # Sources and cell lines
        if "sourceName" in df.columns:
            result["sources"] = df["sourceName"].dropna().unique().tolist()[:3]
        if "sampleName" in df.columns:
            result["n_cell_lines"] = df["sampleName"].nunique()

        return result
    except Exception as e:
        return {"found": False, "error": str(e)[:80]}


def _synergy_to_score(median_zip: float, median_bliss: float) -> float:
    """
    Convert a raw synergy metric to a 0-1 score.
    Uses both ZIP and Bliss, averaged.

    Rules (from Bliss/ZIP interpretation):
      > 10: strong synergy
      5-10: synergy
      0-5:  additive
      < 0:  antagonism
    """
    def scale(v):
        if v is None or pd.isna(v):
            return None
        # Clip to [-20, 20] and map to [0, 1]
        v = max(-20.0, min(20.0, float(v)))
        return (v + 20.0) / 40.0  # 0 at -20, 0.5 at 0, 1.0 at +20

    candidates = [scale(median_zip), scale(median_bliss)]
    candidates = [c for c in candidates if c is not None]
    return float(np.mean(candidates)) if candidates else 0.5


# ==================== Heuristic fallback ====================

def _complementarity_score(corr_value: float) -> float:
    if corr_value < 0.05:
        return 0.30 + corr_value * 4.0
    if corr_value <= 0.50:
        return 0.80
    if corr_value <= 0.80:
        return max(0.20, 0.80 - (corr_value - 0.50) * 2.0)
    return 0.20


# ==================== Main entry ====================

def find_synergy_pairs(candidates_df, corr, top_n=10,
                        check_trials=False, verbose=True):
    """
    Score every pair of candidates for synergy.

    Uses SYNERGxDB real data where available. Falls back to heuristic.
    """
    if candidates_df is None or corr is None or corr.empty:
        return pd.DataFrame()

    drug_names = dict(zip(candidates_df["chembl_id"], candidates_df["drug_name"]))
    ids = [c for c in corr.columns if c in drug_names]

    # Load SYNERGxDB drug list and map our candidates
    if verbose:
        print(f"  Loading SYNERGxDB drug list...")
    try:
        drug_df = _load_drug_list(verbose=verbose)
        # Map each candidate to a SYNERGxDB ID
        cid_to_syn_id = {}
        for cid in ids:
            syn_id = _find_drug_id(drug_names[cid], drug_df)
            if syn_id:
                cid_to_syn_id[cid] = syn_id
        if verbose:
            print(f"  Matched {len(cid_to_syn_id)}/{len(ids)} candidates to SYNERGxDB")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  SYNERGxDB unavailable: {e}")
        drug_df = None
        cid_to_syn_id = {}

    pairs = []
    total = len(ids) * (len(ids) - 1) // 2
    idx = 0

    for a, b in combinations(ids, 2):
        idx += 1
        corr_val = float(corr.loc[a, b])

        # Try real SYNERGxDB data
        real_synergy = None
        n_studies = 0
        sources = []

        if a in cid_to_syn_id and b in cid_to_syn_id:
            result = _query_synergy(cid_to_syn_id[a], cid_to_syn_id[b])
            if result.get("found"):
                zip_med = result.get("zip_median")
                bliss_med = result.get("bliss_median")
                real_synergy = _synergy_to_score(zip_med, bliss_med)
                n_studies = result.get("n_studies", 0)
                sources = result.get("sources", [])
            time.sleep(0.15)  # be polite to the API

        # Heuristic fallback
        comp = _complementarity_score(corr_val)
        prior_bonus = 0.0  # we're not checking trials in this version

        if real_synergy is not None:
            synergy = real_synergy
            source = "synergxdb"
            rationale = f"Real data: {n_studies} studies"
            if sources:
                rationale += f" ({', '.join(sources[:2])})"
            rationale += f" · heuristic r={corr_val:.2f}"
        else:
            synergy = 0.7 * comp + 0.3 * prior_bonus
            source = "heuristic"
            if a in cid_to_syn_id or b in cid_to_syn_id:
                rationale = f"Not in SYNERGxDB · heuristic r={corr_val:.2f}"
            else:
                rationale = f"Neither drug in SYNERGxDB · heuristic r={corr_val:.2f}"

        pairs.append({
            "drug_a":         drug_names[a],
            "drug_b":         drug_names[b],
            "chembl_a":       a,
            "chembl_b":       b,
            "correlation":    round(corr_val, 3),
            "n_studies":      n_studies,
            "synergy_score":  round(synergy, 3),
            "source":         source,
            "rationale":      rationale,
        })

    df = pd.DataFrame(pairs).sort_values("synergy_score", ascending=False).reset_index(drop=True)

    if verbose:
        n_real = (df["source"] == "synergxdb").sum()
        print(f"  Scored {len(df)} pairs | {n_real} with real SYNERGxDB data")

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
        "source": top["source"],
    }
