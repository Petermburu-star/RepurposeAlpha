"""
RepurposeAlpha — disease to candidates discovery via Open Targets.
Ranked selection: best N by clinical phase + approval + drug type.
"""
import requests
import pandas as pd
from typing import Optional

OPEN_TARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"

STAGE_RANK = {
    "APPROVAL":    4.0,
    "PHASE_4":     4.0,
    "PHASE_3":     3.0,
    "PHASE_2_3":   2.5,
    "PHASE_2":     2.0,
    "PHASE_1_2":   1.5,
    "PHASE_1":     1.0,
    "PRECLINICAL": 0.5,
    "UNKNOWN":     0.0,
}


def search_disease(disease_name: str) -> Optional[dict]:
    query = """
    query searchDisease($name: String!) {
      search(queryString: $name, entityNames: ["disease"], page: {index: 0, size: 5}) {
        hits { id name entity }
      }
    }
    """
    try:
        r = requests.post(OPEN_TARGETS_API,
                          json={"query": query, "variables": {"name": disease_name}},
                          timeout=30)
        if r.status_code == 200:
            hits = r.json().get("data", {}).get("search", {}).get("hits", [])
            if hits:
                return hits[0]
    except Exception as e:
        print(f"  Search failed: {e}")
    return None


def get_drugs_for_disease(efo_id: str, verbose: bool = True) -> pd.DataFrame:
    """Fetch drugs via drugAndClinicalCandidates. No isApproved — derived from stage."""
    query = """
    query diseaseDrugs($efoId: String!) {
      disease(efoId: $efoId) {
        id
        name
        drugAndClinicalCandidates {
          count
          rows {
            maxClinicalStage
            drug {
              id
              name
              drugType
            }
          }
        }
      }
    }
    """
    rows = []
    try:
        r = requests.post(OPEN_TARGETS_API,
                          json={"query": query, "variables": {"efoId": efo_id}},
                          timeout=30)
        if r.status_code == 200:
            body = r.json()
            if "errors" in body:
                if verbose:
                    print(f"    GraphQL errors: {body['errors'][:1]}")
            data = body.get("data", {}).get("disease", {})
            if data:
                container = data.get("drugAndClinicalCandidates") or {}
                count = container.get("count", 0)
                if verbose:
                    print(f"    (API reported {count} total associations)")
                for row in container.get("rows", []):
                    drug = row.get("drug", {}) or {}
                    stage = row.get("maxClinicalStage") or "UNKNOWN"
                    rows.append({
                        "chembl_id":   (drug.get("id") or "").replace("CHEMBL_", "CHEMBL"),
                        "drug_name":   drug.get("name", ""),
                        "stage_str":   stage,
                        "max_phase":   STAGE_RANK.get(stage, 0.0),
                        "is_approved": stage in ("APPROVAL", "PHASE_4"),
                        "drug_type":   drug.get("drugType", ""),
                    })
        else:
            if verbose:
                print(f"    HTTP {r.status_code}")
    except Exception as e:
        if verbose:
            print(f"  Query failed: {e}")
    return pd.DataFrame(rows)


def rank_and_select(df: pd.DataFrame, max_results: int,
                    min_phase: int = 2, verbose: bool = True) -> pd.DataFrame:
    """
    Rank and select top N candidates.
    Scoring: 70% phase + 20% approved + 10% biologic.
    """
    if df.empty:
        return df

    df = df.copy()
    df["phase_num"] = df["stage_str"].map(STAGE_RANK).fillna(0.0)

    # Filter by min_phase
    stage_to_phase = {
        "APPROVAL": 4.0, "PHASE_4": 4.0, "PHASE_3": 3.0, "PHASE_2_3": 2.5,
        "PHASE_2": 2.0, "PHASE_1_2": 1.5, "PHASE_1": 1.0, "PRECLINICAL": 0.5,
        "UNKNOWN": 0.0,
    }
    df["phase_for_filter"] = df["stage_str"].map(stage_to_phase).fillna(0.0)
    df = df[df["phase_for_filter"] >= min_phase].drop_duplicates("chembl_id")

    if df.empty:
        return df

    # Scoring
    df["phase_score"] = (df["phase_for_filter"] / 4.0).clip(0, 1)
    df["approved_score"] = df["is_approved"].astype(int)
    df["biologic_score"] = df["drug_type"].str.lower().apply(
        lambda x: 1 if any(k in str(x).lower() for k in ["antibody", "protein", "peptide"]) else 0
    )

    df["rank_score"] = (
        0.70 * df["phase_score"] +
        0.20 * df["approved_score"] +
        0.10 * df["biologic_score"]
    )

    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)

    df["selection_rationale"] = df.apply(
        lambda r: f"Phase {r['stage_str']} · {'approved' if r['is_approved'] else 'investigational'} · score {r['rank_score']:.2f}",
        axis=1,
    )

    if verbose:
        print(f"\n  Ranked {len(df)} candidates. Selection criteria:")
        print(f"    - 70% clinical phase · 20% approved · 10% biologic")

    return df.head(max_results).reset_index(drop=True)


def get_candidates_for_disease(disease_name: str, min_phase: int = 2,
                                max_results: int = 10,
                                verbose: bool = True) -> pd.DataFrame:
    print(f"→ Searching Open Targets for '{disease_name}'...")
    disease = search_disease(disease_name)
    if not disease:
        print(f"  ❌ Not found")
        return pd.DataFrame()

    print(f"  ✓ Found: {disease['name']} (EFO: {disease['id']})")
    print(f"→ Fetching all associated drugs...")
    all_drugs = get_drugs_for_disease(disease["id"], verbose=verbose)

    if all_drugs.empty:
        print(f"  ❌ No drugs returned")
        return pd.DataFrame()

    print(f"  ✓ Retrieved {len(all_drugs)} candidates from Open Targets")

    # Rank by phase + approval + type
    ranked = rank_and_select(all_drugs, max_results=len(all_drugs),
                              min_phase=min_phase, verbose=verbose)

    if ranked.empty:
        return ranked

    # Then select a diverse top-N
    if len(ranked) > max_results:
        print(f"→ Selecting {max_results} most diverse from top candidates...")
        return select_diverse_top_n(ranked, n=max_results, verbose=verbose)
    else:
        return ranked.head(max_results).reset_index(drop=True)


def select_diverse_top_n(df: pd.DataFrame, n: int,
                          rank_col: str = "rank_score",
                          verbose: bool = True) -> pd.DataFrame:
    """
    Rank-first diversity selection.

    1. Sort by rank_score (phase + approved + biologic).
    2. Prefer candidates with drug-name families not already picked.
    3. Fill remaining slots by rank order.

    This avoids 5 variants of the same drug while preserving the
    "most clinically advanced first" ordering.
    """
    if df.empty or len(df) <= n:
        return df

    df = df.copy()
    df["_family"] = df["drug_name"].apply(
        lambda x: str(x).strip().upper().split()[0] if x else "UNKNOWN"
    )

    selected_rows = []
    seen_families = set()
    selected_ids = set()

    # Pass 1: pick one per unique family in rank order
    for _, row in df.iterrows():
        if row["_family"] not in seen_families and row["chembl_id"] not in selected_ids:
            selected_rows.append(row)
            seen_families.add(row["_family"])
            selected_ids.add(row["chembl_id"])
            if len(selected_rows) >= n:
                break

    # Pass 2: fill remaining slots by rank
    if len(selected_rows) < n:
        for _, row in df.iterrows():
            if row["chembl_id"] not in selected_ids:
                selected_rows.append(row)
                selected_ids.add(row["chembl_id"])
                if len(selected_rows) >= n:
                    break

    result = pd.DataFrame(selected_rows).drop(columns=["_family"]).reset_index(drop=True)

    if verbose:
        print(f"  Rank-first diversity: {len(result)} from {len(df)} pool")

    return result

