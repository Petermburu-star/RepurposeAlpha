"""
RepurposeAlpha — FTO screening module (v2).

Simple, honest, differentiable. Uses only what we actually know:
- Approval status (from Open Targets)
- Drug class (small molecule vs biologic)
- Known generic status (a curated list of >20-year-old drugs)

Explicit disclaimer: this is a SCREENING signal, not legal advice.
"""
import pandas as pd
from dataclasses import dataclass
from datetime import datetime


CURRENT_YEAR = datetime.now().year

# Well-known drugs that are almost certainly generic (off-patent, widely available)
# Curated list — a real version would query the FDA Orange Book
KNOWN_GENERIC = {
    "METOPROLOL", "LISINOPRIL", "ATORVASTATIN", "SIMVASTATIN",
    "IBUPROFEN", "ASPIRIN", "METFORMIN", "OMEPRAZOLE",
    "AMOXICILLIN", "AZITHROMYCIN", "WARFARIN", "HEPARIN CALCIUM",
    "ENOXAPARIN SODIUM", "PREDNISOLONE", "DEXAMETHASONE",
    "HYDROCORTISONE", "METHOTREXATE", "GEMFIBROZIL",
    "DIAZOXIDE", "ARGATROBAN",
}

# Well-known biologics (usually harder to genericize, protected longer)
KNOWN_BIOLOGIC = {
    "SOMATROPIN", "INSULIN", "LIRAGLUTIDE", "EXENATIDE",
    "SETMELANOTIDE", "TESAMORELIN", "OXYTOCIN",
    "BEVACIZUMAB", "RITUXIMAB", "TRASTUZUMAB",
}


@dataclass
class FTOAssessment:
    chembl_id: str
    drug_name: str
    status: str
    is_generic: bool
    is_biologic: bool
    fto_risk: str          # "low" | "medium" | "high"
    fto_score: float       # 0.0 - 1.0
    rationale: str


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


def _is_generic(drug_name: str) -> bool:
    if not drug_name:
        return False
    name_upper = drug_name.upper().strip()
    return any(g in name_upper for g in KNOWN_GENERIC)


def _is_biologic(drug_name: str) -> bool:
    if not drug_name:
        return False
    name_upper = drug_name.upper().strip()
    return any(b in name_upper for b in KNOWN_BIOLOGIC)


def assess_candidate(cid, drug_name, stage_str, candidates_df=None):
    status = _stage_to_status(stage_str)
    generic = _is_generic(drug_name)
    biologic = _is_biologic(drug_name)

    # Scoring rules — higher score = better IP position
    if generic:
        score = 0.15
        risk = "high"
        rationale = "Generic drug — no compound patent, weak method-of-use option"
    elif biologic and status != "approved":
        score = 0.75
        risk = "low"
        rationale = "Biologic in development — strong IP, harder to genericize"
    elif biologic and status == "approved":
        score = 0.55
        risk = "medium"
        rationale = "Approved biologic — biosimilar path exists but protected longer"
    elif status == "approved":
        score = 0.40
        risk = "medium"
        rationale = "Approved small molecule — method-of-use patents possible"
    elif status == "phase_3":
        score = 0.65
        risk = "low"
        rationale = "Phase 3 — new-indication IP still fileable"
    elif status == "phase_2":
        score = 0.75
        risk = "low"
        rationale = "Phase 2 — best IP position, no competing filings"
    else:
        score = 0.80
        risk = "low"
        rationale = "Early stage — no IP conflicts yet"

    return FTOAssessment(
        chembl_id=cid,
        drug_name=drug_name,
        status=status,
        is_generic=generic,
        is_biologic=biologic,
        fto_risk=risk,
        fto_score=round(score, 3),
        rationale=rationale,
    )


def assess_portfolio(candidates_df, disease_name=None, verbose=True):
    rows = []
    for _, row in candidates_df.iterrows():
        a = assess_candidate(
            cid=row["chembl_id"],
            drug_name=row.get("drug_name", row["chembl_id"]),
            stage_str=row.get("stage_str", ""),
            candidates_df=candidates_df,
        )
        rows.append({
            "chembl_id":  a.chembl_id,
            "drug_name":  a.drug_name,
            "status":     a.status,
            "generic":    a.is_generic,
            "biologic":   a.is_biologic,
            "fto_risk":   a.fto_risk,
            "fto_score":  a.fto_score,
            "rationale":  a.rationale,
        })
    df = pd.DataFrame(rows).sort_values("fto_score", ascending=False)
    if verbose:
        print(f"  Assessed {len(df)} candidates")
        print(f"  Risk breakdown: {df['fto_risk'].value_counts().to_dict()}")
    return df
