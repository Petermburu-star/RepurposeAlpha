"""
RepurposeAlpha — Cross-disease mechanism discovery (v2, correct schema).
"""
import requests
import pandas as pd
from typing import List

OPEN_TARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"


def get_diseases_for_drug(chembl_id: str, limit: int = 20) -> list:
    """
    Query Open Targets for all diseases associated with a drug.
    Uses the correct schema: indications { rows { disease { name } maxClinicalStage } }
    """
    query = """
    query drugDiseases($chemblId: String!) {
      drug(chemblId: $chemblId) {
        id
        name
        indications {
          count
          rows {
            disease { id name }
            maxClinicalStage
          }
        }
      }
    }
    """
    try:
        r = requests.post(
            OPEN_TARGETS_API,
            json={"query": query, "variables": {"chemblId": chembl_id}},
            timeout=30,
        )
        if r.status_code == 200:
            body = r.json()
            data = body.get("data", {}).get("drug")
            if data and "indications" in data:
                container = data["indications"] or {}
                rows = container.get("rows", [])
                diseases = []
                for row in rows[:limit]:
                    d = row.get("disease") or {}
                    if d.get("name"):
                        diseases.append(d["name"])
                return diseases
            elif "errors" in body:
                # Print first error for debugging
                print(f"    GraphQL errors: {body['errors'][:1]}")
    except Exception as e:
        print(f"    Request failed: {e}")
    return []


def find_cross_disease_overlaps(candidates_df: pd.DataFrame,
                                 current_disease: str,
                                 verbose: bool = True) -> pd.DataFrame:
    """For each candidate, find other diseases it's been tested in."""
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame()

    rows = []
    total = len(candidates_df)
    for i, (_, row) in enumerate(candidates_df.iterrows(), 1):
        cid = row["chembl_id"]
        drug = row.get("drug_name", cid)

        if verbose:
            print(f"  [{i}/{total}] {str(drug)[:40]}...")

        other = get_diseases_for_drug(cid, limit=15)
        other = [d for d in other
                 if current_disease.lower() not in d.lower()
                 and d.lower() not in current_disease.lower()]

        rows.append({
            "chembl_id":          cid,
            "drug_name":          drug,
            "n_other_diseases":   len(other),
            "top_other_disease":  other[0] if other else None,
            "all_other_diseases": "; ".join(other[:5]),
        })

    df = pd.DataFrame(rows).sort_values("n_other_diseases", ascending=False).reset_index(drop=True)

    if verbose:
        n_overlap = (df["n_other_diseases"] > 0).sum()
        print(f"\n  Cross-disease hits: {n_overlap}/{total} candidates")

    return df


def summarize_cross_disease(candidates_df, current_disease, top_n=5):
    df = find_cross_disease_overlaps(candidates_df, current_disease, verbose=False)
    if df.empty:
        return {"summary": "No cross-disease analysis available", "top": []}
    top = df.head(top_n).to_dict("records")
    n_overlap = (df["n_other_diseases"] > 0).sum()
    summary = (f"{n_overlap} of {len(df)} candidates are also in trials for other diseases.")
    return {"summary": summary, "top": top}
