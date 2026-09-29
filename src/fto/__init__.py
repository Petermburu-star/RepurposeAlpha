"""
RepurposeAlpha — Freedom-to-Operate screening (v2 — FDA-backed).

Uses OpenFDA Orange Book + Drugs@FDA APIs for real patent,
exclusivity, and approval data. Falls back to heuristics if
a drug isn't found in the FDA data.

Disclaimer: Screening signal only. Not legal advice.
Consult a patent attorney before making IP decisions.
"""
import requests
import pandas as pd
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

OPENFDA_ORANGE = "https://api.fda.gov/drug/orangebook.json"
OPENFDA_DRUGS  = "https://api.fda.gov/drug/drugsfda.json"

CURRENT_YEAR = datetime.now().year


@dataclass
class FTOAssessment:
    chembl_id: str
    drug_name: str
    status: str
    is_generic: bool
    is_biologic: bool
    fto_risk: str
    fto_score: float
    rationale: str
    n_patents: int = 0
    latest_patent_expiry: str = ""
    n_excl: int = 0
    approval_year: int = None
    data_source: str = "heuristic"    # or "openfda"


# Fallback lists if FDA query fails
KNOWN_GENERIC = {
    "METOPROLOL", "LISINOPRIL", "ATORVASTATIN", "SIMVASTATIN",
    "IBUPROFEN", "ASPIRIN", "METFORMIN", "OMEPRAZOLE",
    "AMOXICILLIN", "AZITHROMYCIN", "WARFARIN", "HEPARIN",
    "ENOXAPARIN", "PREDNISOLONE", "DEXAMETHASONE",
    "HYDROCORTISONE", "METHOTREXATE", "GEMFIBROZIL",
    "DIAZOXIDE", "ARGATROBAN", "PRIMAQUINE", "CHLOROQUINE",
}

KNOWN_BIOLOGIC = {
    "SOMATROPIN", "INSULIN", "LIRAGLUTIDE", "EXENATIDE",
    "SETMELANOTIDE", "TESAMORELIN", "OXYTOCIN",
    "BEVACIZUMAB", "RITUXIMAB", "TRASTUZUMAB", "EPOETIN",
}


def _stage_to_status(stage_str):
    if not stage_str:
        return "phase_2"
    s = str(stage_str).upper().strip().replace(" ", "_")
    if s in ("APPROVAL", "PHASE_4"):
        return "approved"
    if s == "PHASE_3":
        return "phase_3"
    if s in ("PHASE_2", "PHASE_2_3"):
        return "phase_2"
    return "phase_1"


def _is_generic_heuristic(name):
    n = str(name).upper().strip()
    return any(g in n for g in KNOWN_GENERIC)


def _is_biologic_heuristic(name):
    n = str(name).upper().strip()
    return any(b in n for b in KNOWN_BIOLOGIC)


@lru_cache(maxsize=256)
def _query_openfda(drug_name: str) -> dict:
    """
    Query OpenFDA Orange Book for real patent/exclusivity data.
    Returns dict with patents, exclusivity, approval info.
    """
    clean = str(drug_name).upper().strip().split()[0]  # drop salt suffixes
    result = {
        "found": False,
        "n_patents": 0,
        "latest_patent_expiry": "",
        "n_excl": 0,
        "application_number": "",
        "sponsor": "",
        "error": None,
    }

    try:
        r = requests.get(
            OPENFDA_ORANGE,
            params={"search": f'products.active_ingredients.name:"{clean}"', "limit": 5},
            timeout=20,
        )
        if r.status_code != 200:
            result["error"] = f"HTTP {r.status_code}"
            return result

        data = r.json()
        records = data.get("results", [])
        if not records:
            return result

        result["found"] = True

        # Aggregate patents and exclusivity across all matching records
        all_patents = []
        all_excl = []
        for rec in records:
            for p in rec.get("patents", []):
                exp = p.get("expiration_date", "")
                if exp:
                    all_patents.append(exp)
            for e in rec.get("exclusivity", []):
                exp = e.get("exclusivity_expiration_date", "")
                if exp:
                    all_excl.append(exp)

        result["n_patents"] = len(all_patents)
        result["latest_patent_expiry"] = max(all_patents) if all_patents else ""
        result["n_excl"] = len(all_excl)

        # Application number from first record
        if records:
            rec = records[0]
            result["application_number"] = rec.get("application_number", "")
            result["sponsor"] = rec.get("sponsor_name", "")

    except Exception as e:
        result["error"] = str(e)[:80]

    return result


def assess_candidate(cid, drug_name, stage_str, candidates_df=None):
    """Assess FTO for one candidate using real FDA data when available."""
    status = _stage_to_status(stage_str)
    drug_upper = str(drug_name).upper().strip()

    # Query FDA
    fda = _query_openfda(drug_upper)

    n_patents = fda.get("n_patents", 0)
    n_excl = fda.get("n_excl", 0)
    latest_expiry = fda.get("latest_patent_expiry", "")

    # Determine protection status
    has_active_patent = False
    if latest_expiry:
        try:
            expiry_year = int(latest_expiry[:4])
            has_active_patent = expiry_year >= CURRENT_YEAR
        except Exception:
            pass

    has_exclusivity = n_excl > 0

    # ---------- Scoring ----------
    if fda["found"]:
        # REAL FDA DATA
        data_source = "openfda"
        if has_active_patent and has_exclusivity:
            score, risk = 0.85, "low"
            rationale = f"Active patent until {latest_expiry} · {n_excl} exclusivity period(s)"
        elif has_active_patent:
            score, risk = 0.70, "low"
            rationale = f"Active patent until {latest_expiry}"
        elif has_exclusivity:
            score, risk = 0.60, "medium"
            rationale = f"{n_excl} exclusivity period(s) — no active patents"
        elif n_patents == 0 and n_excl == 0:
            # No patents + no exclusivity = generic regardless of current trial phase
            score, risk = 0.15, "high"
            rationale = "No active patents or exclusivity — generic, no protection possible"
        elif status == "approved":
            score, risk = 0.30, "medium"
            rationale = f"{n_patents} expired patent(s) — limited protection"
        else:
            score, risk = 0.65, "low"
            rationale = "Investigational — new-indication IP still fileable"
    else:
        # FALLBACK HEURISTIC
        data_source = "heuristic"
        is_generic = _is_generic_heuristic(drug_name)
        is_biologic = _is_biologic_heuristic(drug_name)

        if is_generic:
            score, risk = 0.15, "high"
            rationale = "Not in FDA data · likely generic (heuristic)"
        elif is_biologic:
            score, risk = 0.70, "low"
            rationale = "Biologic (heuristic) — harder to genericize"
        elif status == "approved":
            score, risk = 0.40, "medium"
            rationale = "Approved small molecule · method-of-use patents possible"
        elif status == "phase_3":
            score, risk = 0.65, "low"
            rationale = "Phase 3 — new-indication IP fileable"
        else:
            score, risk = 0.75, "low"
            rationale = "Early stage — no IP conflicts"

    is_generic = (risk == "high" and n_patents == 0 and n_excl == 0)
    is_biologic = _is_biologic_heuristic(drug_name)

    return FTOAssessment(
        chembl_id=cid,
        drug_name=drug_name,
        status=status,
        is_generic=is_generic,
        is_biologic=is_biologic,
        fto_risk=risk,
        fto_score=round(score, 3),
        rationale=rationale,
        n_patents=n_patents,
        latest_patent_expiry=latest_expiry,
        n_excl=n_excl,
        data_source=data_source,
    )


def assess_portfolio(candidates_df, disease_name=None, verbose=True):
    """Run FTO assessment across all candidates."""
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame()

    rows = []
    total = len(candidates_df)
    n_fda = 0

    for i, (_, row) in enumerate(candidates_df.iterrows(), 1):
        drug = row.get("drug_name", row["chembl_id"])
        if verbose:
            print(f"  [{i}/{total}] {str(drug)[:40]}...")

        a = assess_candidate(
            cid=row["chembl_id"],
            drug_name=drug,
            stage_str=row.get("stage_str", ""),
            candidates_df=candidates_df,
        )
        if a.data_source == "openfda":
            n_fda += 1

        rows.append({
            "chembl_id":     a.chembl_id,
            "drug_name":     a.drug_name,
            "status":        a.status,
            "fto_risk":      a.fto_risk,
            "fto_score":     a.fto_score,
            "n_patents":     a.n_patents,
            "latest_patent": a.latest_patent_expiry,
            "n_excl":        a.n_excl,
            "source":        a.data_source,
            "rationale":     a.rationale,
        })

    df = pd.DataFrame(rows).sort_values("fto_score", ascending=False).reset_index(drop=True)

    if verbose:
        counts = df["fto_risk"].value_counts().to_dict()
        print(f"\n  Risk breakdown: {counts}")
        print(f"  FDA data available: {n_fda}/{total}")

    return df
