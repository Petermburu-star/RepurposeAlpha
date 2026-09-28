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
