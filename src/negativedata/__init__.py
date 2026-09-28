"""
RepurposeAlpha — Negative results intelligence.

Distinguishes failure types:
- FUTILITY / SAFETY → strong negative signal (skip)
- ENROLLMENT / BUSINESS → weak signal (consider retrying)
- Weights recent failures more heavily (5-year half-life)
"""
import requests
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import List


CT_BASE = "https://clinicaltrials.gov/api/v2/studies"
CURRENT_YEAR = datetime.now().year


@dataclass
class FailedTrial:
    nct_id: str
    title: str
    status: str
    reason: str
    phase: str
    condition: str
    start_year: int


FAILURE_STRENGTH = {
    "FUTILITY":   1.0,
    "SAFETY":     1.0,
    "ENROLLMENT": 0.3,
    "BUSINESS":   0.2,
    "UNKNOWN":    0.5,
}


def _classify_reason(why_text: str) -> str:
    w = (why_text or "").lower()
    if any(k in w for k in ("futility", "efficacy", "ineffective", "no benefit", "did not meet")):
        return "FUTILITY"
    if any(k in w for k in ("safety", "adverse", "toxicity", "side effect")):
        return "SAFETY"
    if any(k in w for k in ("enroll", "accrual", "recruit", "slow")):
        return "ENROLLMENT"
    if any(k in w for k in ("business", "strategic", "sponsor", "funding", "commercial")):
        return "BUSINESS"
    return "UNKNOWN"


def search_failed_trials(drug_name: str, limit: int = 50) -> List[FailedTrial]:
    if not drug_name:
        return []
    params = {"query.intr": drug_name, "pageSize": limit, "format": "json"}
    try:
        r = requests.get(CT_BASE, params=params, timeout=30)
        if r.status_code != 200:
            return []
        studies = r.json().get("studies", [])
    except Exception:
        return []

    results = []
    for s in studies:
        protocol   = s.get("protocolSection", {})
        status_mod = protocol.get("statusModule", {})
        design_mod = protocol.get("designModule", {})
        ident_mod  = protocol.get("identificationModule", {})
        cond_mod   = protocol.get("conditionsModule", {})

        status = (status_mod.get("overallStatus") or "").upper()
        if status not in ("TERMINATED", "WITHDRAWN", "SUSPENDED"):
            continue

        reason = _classify_reason(status_mod.get("whyStopped"))
        start_date = status_mod.get("startDateStruct", {}).get("date") or ""
        start_year = int(start_date[:4]) if len(start_date) >= 4 and start_date[:4].isdigit() else None

        results.append(FailedTrial(
            nct_id=ident_mod.get("nctId", ""),
            title=(ident_mod.get("briefTitle") or "")[:120],
            status=status,
            reason=reason,
            phase=", ".join(design_mod.get("phases", [])) or "N/A",
            condition=", ".join(cond_mod.get("conditions", []))[:120],
            start_year=start_year,
        ))

    return results


def _decay_weight(year: int) -> float:
    if not year:
        return 0.5
    age = max(0, CURRENT_YEAR - year)
    return 0.5 ** (age / 5.0)


def analyze_candidate_failures(drug_name: str, current_disease: str = None) -> dict:
    trials = search_failed_trials(drug_name)
    if not trials:
        return {
            "n_failed": 0, "n_relevant": 0, "reasons": {},
            "weighted_risk": 0.0, "risk_flag": "clean",
            "trials": [], "message": "No terminated trials found.",
        }

    weighted = 0.0
    reasons = {}
    relevant = 0
    for t in trials:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
        strength = FAILURE_STRENGTH.get(t.reason, 0.5)
        decay = _decay_weight(t.start_year)
        weighted += strength * decay

        if current_disease:
            cf = current_disease.lower()
            if cf in t.condition.lower() or t.condition.lower() in cf:
                relevant += 1

    if weighted >= 2.0:
        risk = "high"
    elif weighted >= 1.0:
        risk = "medium"
    elif weighted >= 0.3:
        risk = "low"
    else:
        risk = "clean"

    parts = []
    for label in ["FUTILITY", "SAFETY", "ENROLLMENT", "BUSINESS", "UNKNOWN"]:
        if reasons.get(label):
            parts.append(f"{reasons[label]} {label.lower()}")
    message = ", ".join(parts) if parts else "no failures"

    return {
        "n_failed":      len(trials),
        "n_relevant":    relevant,
        "reasons":       reasons,
        "weighted_risk": round(weighted, 2),
        "risk_flag":     risk,
        "trials":        trials,
        "message":       message,
    }


def analyze_portfolio(candidates_df: pd.DataFrame, disease: str = None,
                       verbose: bool = True) -> pd.DataFrame:
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame()

    rows = []
    total = len(candidates_df)
    for i, (_, row) in enumerate(candidates_df.iterrows(), 1):
        drug = row.get("drug_name", row["chembl_id"])
        if verbose:
            print(f"  [{i}/{total}] {str(drug)[:40]}...")

        summary = analyze_candidate_failures(drug, current_disease=disease)
        rows.append({
            "chembl_id":     row["chembl_id"],
            "drug_name":     drug,
            "n_failed":      summary["n_failed"],
            "n_relevant":    summary["n_relevant"],
            "risk_flag":     summary["risk_flag"],
            "weighted_risk": summary["weighted_risk"],
            "breakdown":     summary["message"],
        })

    df = pd.DataFrame(rows)
    order = {"high": 0, "medium": 1, "low": 2, "clean": 3}
    df["_order"] = df["risk_flag"].map(order)
    df = df.sort_values(["_order", "weighted_risk"], ascending=[True, False]).drop(columns=["_order"])

    if verbose:
        print(f"\n  Risk breakdown: {df['risk_flag'].value_counts().to_dict()}")

    return df.reset_index(drop=True)
