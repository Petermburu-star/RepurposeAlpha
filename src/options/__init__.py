"""
RepurposeAlpha — Real Options Valuation engine v3.

Correctly:
- Deducts phase costs inside the lattice
- Uses EXPECTED value at each gate (not best-case path)
- Evaluates each gate as a fresh sub-tree (correct "continue or abandon" logic)
"""
import numpy as np
from math import comb
from dataclasses import dataclass
from typing import List


@dataclass
class Phase:
    name: str
    duration_years: float
    cost: float
    prob_success: float


@dataclass
class OptionResult:
    asset_value: float
    option_value: float
    abandonment_value: float
    decision_tree: List[dict]
    threshold_npv: float


class RealOptionsEngine:
    def __init__(self, phases, peak_sales, volatility, risk_free_rate,
                 discount_rate, profit_margin=0.30):
        self.phases = phases
        self.peak_sales = peak_sales
        self.volatility = volatility
        self.r = risk_free_rate
        self.discount_rate = discount_rate
        self.margin = profit_margin

    def _phase_step_ranges(self, n_steps):
        """Return a list of (start_step, end_step) for each phase."""
        total_years = sum(ph.duration_years for ph in self.phases)
        ranges = []
        step = 0
        for ph in self.phases:
            phase_steps = max(1, int(round(ph.duration_years / total_years * n_steps)))
            end = min(step + phase_steps, n_steps)
            ranges.append((step, end))
            step = end
        return ranges

    def _build_lattice(self, n_steps):
        dt = 1.0
        u = np.exp(self.volatility * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        S = np.zeros((n_steps + 1, n_steps + 1))
        for i in range(n_steps + 1):
            for j in range(i + 1):
                S[j, i] = self.peak_sales * (u ** (i - j)) * (d ** j)
        return S, p, u, d

    def _backward_induction(self, S, p, n_steps, cost_by_step, start_step=0):
        """
        Compute V for a sub-tree starting at start_step.
        Only costs from start_step onwards are deducted.
        """
        V = np.maximum(S * self.margin, 0.0)
        for i in range(n_steps - 1, start_step - 1, -1):
            cost = cost_by_step.get(i, 0.0)
            for j in range(i + 1):
                cont = np.exp(-self.r) * (p * V[j, i + 1] + (1 - p) * V[j + 1, i + 1])
                V[j, i] = max(cont - cost, 0.0)
        return V

    def _expected_at_step(self, V, p, step):
        """Risk-neutral expected value at a given step across all nodes."""
        if step > V.shape[1] - 1:
            return 0.0
        total = 0.0
        for j in range(step + 1):
            prob = comb(step, j) * (p ** (step - j)) * ((1 - p) ** j)
            total += prob * V[j, step]
        return float(total)

    def value(self, n_steps=10):
        S, p, u, d = self._build_lattice(n_steps)

        # Full-tree option value (starting from now)
        full_cost = {}
        ranges = self._phase_step_ranges(n_steps)
        for (s, e), ph in zip(ranges, self.phases):
            per_step = ph.cost / max(1, (e - s))
            for k in range(s, e):
                full_cost[k] = full_cost.get(k, 0.0) + per_step

        V_full = self._backward_induction(S, p, n_steps, full_cost, start_step=0)
        option_val = float(V_full[0, 0])

        # Naive DCF
        total_cost = sum(ph.cost for ph in self.phases)
        total_duration = sum(ph.duration_years for ph in self.phases)
        prob_all_success = float(np.prod([ph.prob_success for ph in self.phases]))
        expected_revenue = (prob_all_success * self.peak_sales * self.margin
                            * np.exp(-self.discount_rate * total_duration))
        dcf_value = float(expected_revenue - total_cost)

        # Per-phase: sub-tree from each gate onwards
        decisions = []
        cumulative_cost = 0.0
        cumulative_prob = 1.0
        for idx, ph in enumerate(self.phases):
            cumulative_cost += ph.cost
            cumulative_prob *= ph.prob_success

            gate_step = ranges[idx][0]

            # Sub-tree from this gate: only remaining costs
            sub_cost = {k: v for k, v in full_cost.items() if k >= gate_step}
            V_sub = self._backward_induction(S, p, n_steps, sub_cost, start_step=gate_step)
            # Expected value at gate = V at gate_step, top node is fine here since
            # we are evaluating "if we are at gate, what is continuation value"
            sub_value = self._expected_at_step(V_sub, p, gate_step)

            decisions.append({
                "phase": ph.name,
                "cumulative_cost": round(cumulative_cost / 1e6, 1),
                "prob_success": round(cumulative_prob, 3),
                "continue": bool(sub_value > 0),
                "option_value_at_gate": round(sub_value / 1e6, 2),
            })

        return OptionResult(
            asset_value=round(dcf_value / 1e6, 2),
            option_value=round(option_val / 1e6, 2),
            abandonment_value=round((option_val - dcf_value) / 1e6, 2),
            decision_tree=decisions,
            threshold_npv=0.0,
        )


# ============================================================
# Per-candidate Real Options analysis
# ============================================================

# What phases remain, given the candidate's current phase
PHASE_PROGRESSION = {
    "approval":    ["Phase 3 (repurposing)", "NDA"],
    "phase_4":     ["Phase 4 (post-marketing)"],
    "phase_3":     ["Phase 3", "NDA"],
    "phase_2_3":   ["Phase 3", "NDA"],
    "phase_2":     ["Phase 2", "Phase 3", "NDA"],
    "phase_1_2":   ["Phase 2", "Phase 3", "NDA"],
    "phase_1":     ["Phase 1", "Phase 2", "Phase 3", "NDA"],
    "preclinical": ["Phase 1", "Phase 2", "Phase 3", "NDA"],
}

# Standard phase parameters
PHASE_COSTS = {
    "Phase 1": 10e6,
    "Phase 2": 25e6,
    "Phase 3": 100e6,
    "Phase 3 (repurposing)": 50e6,   # cheaper than full Phase 3
    "Phase 4 (post-marketing)": 5e6,
    "NDA": 5e6,
}

PHASE_POS = {
    "Phase 1": 0.63,
    "Phase 2": 0.31,
    "Phase 3": 0.58,
    "Phase 3 (repurposing)": 0.65,   # higher PoS — different indication, but safety known
    "Phase 4 (post-marketing)": 0.85,
    "NDA": 0.85,
}

PHASE_DURATION = {
    "Phase 1": 2.0,
    "Phase 2": 2.0,
    "Phase 3": 3.0,
    "Phase 3 (repurposing)": 2.0,
    "Phase 4 (post-marketing)": 1.0,
    "NDA": 1.0,
}


def _stage_to_key(stage_str):
    if not stage_str:
        return "phase_2"
    s = str(stage_str).upper().strip().replace(" ", "_")
    mapping = {
        "APPROVAL": "approval", "PHASE_4": "phase_4", "PHASE_3": "phase_3",
        "PHASE_2_3": "phase_2_3", "PHASE_2": "phase_2",
        "PHASE_1_2": "phase_1_2", "PHASE_1": "phase_1", "PRECLINICAL": "preclinical",
    }
    return mapping.get(s, "phase_2")


def assess_candidate_option(stage_str, peak_sales, volatility, rfr,
                              discount_rate=0.10, profit_margin=0.30):
    """
    Run Real Options for a SINGLE candidate given its current stage.
    Returns dict with option_value, decision, and phases used.
    """
    phase_key = _stage_to_key(stage_str)
    remaining_phases = PHASE_PROGRESSION.get(phase_key, ["Phase 3", "NDA"])

    phases = []
    for pname in remaining_phases:
        phases.append(Phase(
            name=pname,
            duration_years=PHASE_DURATION.get(pname, 2.0),
            cost=PHASE_COSTS.get(pname, 25e6),
            prob_success=PHASE_POS.get(pname, 0.5),
        ))

    engine = RealOptionsEngine(
        phases=phases,
        peak_sales=peak_sales,
        volatility=volatility,
        risk_free_rate=rfr,
        discount_rate=discount_rate,
        profit_margin=profit_margin,
    )
    result = engine.value()

    decision = "CONTINUE" if result.option_value > 0 else "ABANDON"

    return {
        "phase_key": phase_key,
        "remaining_phases": remaining_phases,
        "option_value": result.option_value,
        "asset_value": result.asset_value,
        "flexibility_premium": result.abandonment_value,
        "decision": decision,
    }


def assess_portfolio_options(candidates_df, corr, disease, rfr,
                               fto_df=None, priors_df=None, verbose=True):
    """
    Run Real Options for each candidate.

    Adjusts peak sales by FTO status:
      - Generic (high risk, no patents) → 20% of base peak sales (no commercial path)
      - Medium FTO risk → 60% of base peak sales
      - Low FTO risk → 100% of base peak sales

    Adjusts PoS by prior failures:
      - high prior risk → multiply PoS by 0.5
      - medium prior risk → multiply PoS by 0.75
    """
    import pandas as pd

    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame()

    base_peak = _estimate_peak_sales(disease)
    base_vol = _estimate_volatility(disease)

    # Build lookups
    fto_lookup = {}
    if fto_df is not None and not fto_df.empty:
        for _, r in fto_df.iterrows():
            fto_lookup[r["chembl_id"]] = {
                "risk": r.get("fto_risk", "medium"),
                "source": r.get("source", "heuristic"),
                "n_patents": r.get("n_patents", 0),
            }

    priors_lookup = {}
    if priors_df is not None and not priors_df.empty:
        for _, r in priors_df.iterrows():
            priors_lookup[r["chembl_id"]] = r.get("risk_flag", "clean")

    rows = []
    for _, row in candidates_df.iterrows():
        cid = row["chembl_id"]
        drug = row.get("drug_name", cid)

        # Adjust peak sales by FTO
        fto_info = fto_lookup.get(cid, {})
        fto_risk = fto_info.get("risk", "medium")
        if fto_risk == "high":
            peak = base_peak * 0.20   # generic — hard to commercialize
            fto_mult = 0.20
        elif fto_risk == "medium":
            peak = base_peak * 0.60
            fto_mult = 0.60
        else:
            peak = base_peak
            fto_mult = 1.00

        # Adjust PoS by prior results
        prior_risk = priors_lookup.get(cid, "clean")
        if prior_risk == "high":
            pos_mult = 0.50
        elif prior_risk == "medium":
            pos_mult = 0.75
        else:
            pos_mult = 1.00

        # Run engine with adjusted inputs
        phase_key = _stage_to_key(row.get("stage_str", ""))
        remaining = PHASE_PROGRESSION.get(phase_key, ["Phase 3", "NDA"])

        phases = []
        for pname in remaining:
            base_pos = PHASE_POS.get(pname, 0.5)
            adjusted_pos = base_pos * pos_mult
            phases.append(Phase(
                name=pname,
                duration_years=PHASE_DURATION.get(pname, 2.0),
                cost=PHASE_COSTS.get(pname, 25e6),
                prob_success=adjusted_pos,
            ))

        try:
            engine = RealOptionsEngine(
                phases=phases,
                peak_sales=peak,
                volatility=base_vol,
                risk_free_rate=rfr,
                discount_rate=0.10,
                profit_margin=0.30,
            )
            result = engine.value()
            ov = result.option_value
            av = result.asset_value
            fp = result.abandonment_value
        except Exception as e:
            if verbose:
                print(f"  ⚠️  {drug}: {e}")
            ov = av = fp = 0.0

        # Priority tiering: CONTINUE has 3 sub-tiers based on option value
        if ov <= 0:
            decision = "ABANDON"
            tier = "Do not pursue"
        elif ov < 50:
            decision = "CONTINUE"
            tier = "Low priority"
        elif ov < 150:
            decision = "CONTINUE"
            tier = "Medium priority"
        else:
            decision = "CONTINUE"
            tier = "High priority"

        # Reason string
        reasons = []
        if fto_mult < 1.0:
            reasons.append(f"FTO {fto_risk} (peak×{fto_mult})")
        if pos_mult < 1.0:
            reasons.append(f"prior {prior_risk} (PoS×{pos_mult})")
        reason = " · ".join(reasons) if reasons else "clean profile"

        rows.append({
            "chembl_id":       cid,
            "drug_name":       drug,
            "current_stage":   row.get("stage_str", ""),
            "option_value_m":  round(ov, 1),
            "asset_value_m":   round(av, 1),
            "flexibility_m":   round(fp, 1),
            "decision":        decision,
            "tier":            tier,
            "adjustment":      reason,
        })

    df = pd.DataFrame(rows).sort_values("option_value_m", ascending=False).reset_index(drop=True)

    if verbose:
        n_cont = int((df["decision"] == "CONTINUE").sum())
        n_aband = len(df) - n_cont
        print(f"  Assessed {len(df)} candidates | {n_cont} CONTINUE, {n_aband} ABANDON")

    return df


def _estimate_peak_sales(disease_name):
    """Local copy to avoid circular import."""
    if not disease_name:
        return 200e6
    d = disease_name.lower()
    if any(k in d for k in ["syndrome", "dystrophy", "deficiency", "gaucher", "fabry"]):
        return 500e6
    if any(k in d for k in ["malaria", "tuberculosis", "leishmaniasis", "schistosomiasis",
                             "trypanosomiasis", "chagas", "dengue", "ebola"]):
        return 300e6
    if any(k in d for k in ["cancer", "carcinoma", "leukemia", "lymphoma", "melanoma", "tumor"]):
        return 800e6
    if any(k in d for k in ["diabetes", "hypertension", "cardiovascular", "coronary", "asthma"]):
        return 1000e6
    return 200e6


def _estimate_volatility(disease_name):
    if not disease_name:
        return 0.40
    d = disease_name.lower()
    if any(k in d for k in ["cancer", "oncology", "leukemia", "lymphoma"]):
        return 0.55
    if any(k in d for k in ["malaria", "tuberculosis", "infectious"]):
        return 0.35
    if any(k in d for k in ["syndrome", "dystrophy", "rare"]):
        return 0.45
    return 0.40
