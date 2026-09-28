"""
RepurposeAlpha — Cost-based pricing framework.

Estimates the sustainable price per patient per year for a repurposed drug.
Uses a "cost-plus" model that ensures development costs are recoverable
within a reasonable timeframe.

Formula:
    required_price = (total_dev_cost / target_patients) * (1 + target_return)
                   / years_on_market

Where:
    total_dev_cost = cumulative phase costs
    target_patients = prevalence × expected market share
    target_return = annual return expected by investors (WACC-based)
"""
import pandas as pd
from dataclasses import dataclass


@dataclass
class PricingResult:
    drug_name: str
    phase_key: str
    estimated_dev_cost_m: float
    target_patients: int
    years_to_recover: int
    sustainable_price_usd: float
    market_reference_price: float
    price_headroom_pct: float
    viability: str    # "commercially_viable" | "marginal" | "not_viable"


# Assumption defaults (would come from registry in production)
DEFAULT_WACC = 0.10            # 10% cost of capital
DEFAULT_YEARS_TO_RECOVER = 7   # 7 years to recover development cost
DEFAULT_MARKET_SHARE = 0.15    # 15% market penetration


# Reference prices per year for common rare/neglected diseases (USD)
# Sources: public pricing databases, WHO/HAI data
MARKET_REFERENCE_PRICES = {
    "rare_disease": 150_000,
    "orphan": 100_000,
    "neglected_tropical": 500,
    "common_disease": 5_000,
    "oncology": 120_000,
    "infectious": 2_000,
}


def _classify_disease_market(disease_name: str) -> tuple:
    """Return (market_category, estimated_prevalence) for the disease."""
    if not disease_name:
        return "common_disease", 1_000_000
    d = disease_name.lower()

    # Rare / orphan indicators
    rare_keywords = ["syndrome", "dystrophy", "deficiency", "rare", "orphan",
                     "gaucher", "fabry", "pompe", "huntington"]
    ntd_keywords = ["malaria", "tuberculosis", "leishmaniasis", "schistosomiasis",
                    "trypanosomiasis", "chagas", "elephantiasis", "trachoma",
                    "leprosy", "dengue"]
    onco_keywords = ["cancer", "carcinoma", "leukemia", "lymphoma", "melanoma", "tumor"]

    if any(k in d for k in rare_keywords):
        return "rare_disease", 15_000
    if any(k in d for k in ntd_keywords):
        return "neglected_tropical", 5_000_000
    if any(k in d for k in onco_keywords):
        return "oncology", 200_000
    return "common_disease", 1_000_000


def estimate_sustainable_price(drug_name, phase_key, disease_name,
                                dev_cost_usd, wacc=DEFAULT_WACC,
                                years=DEFAULT_YEARS_TO_RECOVER,
                                market_share=DEFAULT_MARKET_SHARE):
    """Estimate the sustainable price per patient per year."""

    market_cat, prevalence = _classify_disease_market(disease_name)
    target_patients = int(prevalence * market_share)

    # Required revenue to recover dev cost + return
    required_revenue = dev_cost_usd * (1 + wacc * years)

    # Price per patient per year
    if target_patients > 0 and years > 0:
        sustainable_price = required_revenue / (target_patients * years)
    else:
        sustainable_price = 0.0

    # Market reference
    reference_price = MARKET_REFERENCE_PRICES.get(market_cat, 5_000)

    # Viability assessment
    if sustainable_price <= reference_price:
        viability = "commercially_viable"
        gap = (reference_price - sustainable_price) / reference_price * 100
    elif sustainable_price <= reference_price * 1.5:
        viability = "marginal"
        gap = (sustainable_price - reference_price) / reference_price * 100
    else:
        viability = "not_viable"
        gap = (sustainable_price - reference_price) / reference_price * 100

    return PricingResult(
        drug_name=drug_name,
        phase_key=phase_key,
        estimated_dev_cost_m=round(dev_cost_usd / 1e6, 1),
        target_patients=target_patients,
        years_to_recover=years,
        sustainable_price_usd=round(sustainable_price, 0),
        market_reference_price=round(reference_price, 0),
        price_headroom_pct=round(gap, 1),
        viability=viability,
    )


def analyze_portfolio(candidates_df, disease, verbose=True):
    """Run pricing analysis across all candidates."""
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame()

    # Import assumptions for phase costs
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import assumptions

    # Registry phase cost mapping
    PHASE_KEY_TO_COST = {
        "approval":    "cost.phase_3_usd",
        "phase_4":     "cost.phase_3_usd",
        "phase_3":     "cost.phase_3_usd",
        "phase_2_3":   "cost.phase_2_3_usd",
        "phase_2":     "cost.phase_2_usd",
        "phase_1_2":   "cost.phase_1_2_usd",
        "phase_1":     "cost.phase_1_usd",
        "preclinical": "cost.phase_0_usd",
    }

    def stage_key(s):
        if not s:
            return "phase_2"
        s = str(s).upper().replace(" ", "_")
        mapping = {
            "APPROVAL": "approval", "PHASE_4": "phase_4", "PHASE_3": "phase_3",
            "PHASE_2_3": "phase_2_3", "PHASE_2": "phase_2",
            "PHASE_1_2": "phase_1_2", "PHASE_1": "phase_1",
        }
        return mapping.get(s, "phase_2")

    rows = []
    for _, row in candidates_df.iterrows():
        drug = row.get("drug_name", row["chembl_id"])
        phase = stage_key(row.get("stage_str", ""))

        cost_key = PHASE_KEY_TO_COST.get(phase, "cost.phase_2_usd")
        dev_cost = float(assumptions.get_value(cost_key, 25e6))

        result = estimate_sustainable_price(drug, phase, disease, dev_cost)
        rows.append({
            "drug_name":       result.drug_name,
            "phase":           result.phase_key,
            "dev_cost_m":      result.estimated_dev_cost_m,
            "patients":        result.target_patients,
            "sustainable_usd": result.sustainable_price_usd,
            "reference_usd":   result.market_reference_price,
            "headroom_pct":         result.price_headroom_pct,
            "viability":       result.viability,
        })

    df = pd.DataFrame(rows)
    # Sort: viable first
    order = {"commercially_viable": 0, "marginal": 1, "not_viable": 2}
    df["_o"] = df["viability"].map(order)
    df = df.sort_values("_o").drop(columns=["_o"]).reset_index(drop=True)

    if verbose:
        counts = df["viability"].value_counts().to_dict()
        print(f"  Viability: {counts}")

    return df
