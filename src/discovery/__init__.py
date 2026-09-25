"""
RepurposeAlpha — disease to candidates discovery via Open Targets.
Maps maxClinicalStage enum → numeric phase.
"""
import requests
import pandas as pd
from typing import Optional

OPEN_TARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"

# Open Targets clinical stage enum → numeric phase
STAGE_TO_PHASE = {
    "APPROVAL":        4.0,
    "PHASE_4":         4.0,
    "PHASE_3":         3.0,
    "PHASE_2_3":       2.5,
    "PHASE_2":         2.0,
    "PHASE_1_2":       1.5,
    "PHASE_1":         1.0,
    "PHASE_0":         0.0,
    "PRECLINICAL":    -1.0,
    "UNKNOWN":         None,
    "WITHDRAWN":       None,
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


def get_drugs_for_disease(efo_id: str) -> pd.DataFrame:
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
              maximumClinicalStage
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
            data = body.get("data", {}).get("disease", {})
            if data:
                container = data.get("drugAndClinicalCandidates") or {}
                count = container.get("count", 0)
                for row in container.get("rows", []):
                    drug = row.get("drug", {}) or {}
                    stage = row.get("maxClinicalStage")
                    rows.append({
                        "chembl_id": (drug.get("id") or "").replace("CHEMBL_", "CHEMBL"),
                        "drug_name": drug.get("name", ""),
                        "stage_str": stage,
                        "max_phase": STAGE_TO_PHASE.get(stage),
                        "drug_type": drug.get("drugType", ""),
                    })
                print(f"    (API reported count: {count})")
    except Exception as e:
        print(f"  Drug query failed: {e}")
    return pd.DataFrame(rows)


def get_candidates_for_disease(disease_name: str, min_phase: float = 2,
                                max_results: int = 50) -> pd.DataFrame:
    print(f"→ Searching Open Targets for '{disease_name}'...")
    disease = search_disease(disease_name)
    if not disease:
        print(f"  ❌ Not found")
        return pd.DataFrame()

    print(f"  ✓ Found: {disease['name']} (EFO: {disease['id']})")
    print(f"→ Fetching drugs via drugAndClinicalCandidates...")
    drugs = get_drugs_for_disease(disease["id"])
    if drugs.empty:
        print(f"  ❌ No drugs returned")
        return pd.DataFrame()

    print(f"  ✓ Found {len(drugs)} drug-disease associations")

    # Show stage distribution for transparency
    print(f"    Stage distribution:")
    for stage, n in drugs["stage_str"].value_counts().items():
        print(f"      {stage or 'null':15s} {n}")

    # Filter: keep rows with max_phase >= min_phase
    filtered = drugs[drugs["max_phase"].notna() & (drugs["max_phase"] >= min_phase)]
    filtered = filtered.drop_duplicates("chembl_id").head(max_results)

    print(f"  After filtering (phase >= {min_phase}): {len(filtered)} unique drugs")
    return filtered.reset_index(drop=True)
