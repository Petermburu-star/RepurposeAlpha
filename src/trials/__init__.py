"""
RepurposeAlpha — Adaptive trial design recommender.

Given a set of candidate drugs to test, recommends the trial architecture
that minimizes time and cost while maintaining statistical power:

1. Fixed 2-arm RCT (standard)
2. Bayesian adaptive (response-adaptive randomization)
3. Platform trial (multi-arm, add/drop)

Key metric: expected sample size reduction vs. running N independent trials.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats


@dataclass
class TrialDesign:
    name: str
    n_arms: int
    expected_sample_size: int
    expected_duration_months: float
    expected_cost_usd: float
    description: str


# Standard assumptions
DEFAULT_EFFECT_SIZE = 0.25       # Cohen's h — moderate effect
DEFAULT_ALPHA = 0.05             # Type I error
DEFAULT_POWER = 0.80             # 1 - Type II error
COST_PER_PATIENT = 25_000        # USD
MONTHLY_ENROLLMENT = 20          # patients/month per arm (assumption)


def _fixed_trial_sample_size(effect_size, alpha=0.05, power=0.80):
    """Standard 2-sample z-test sample size calculation."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n_per_arm = ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n_per_arm * 2))  # total


def _adaptive_efficiency_ratio(n_arms):
    """
    Bayesian adaptive designs reduce sample size through:
    - Early stopping for futility/success
    - Allocation shifting toward effective arms
    Empirical studies show 25-40% reduction for 3-4 arm trials.
    """
    if n_arms <= 1:
        return 1.0
    elif n_arms == 2:
        return 0.85       # 15% reduction
    elif n_arms <= 4:
        return 0.70       # 30% reduction
    else:
        return 0.60       # 40% reduction


def _platform_efficiency_ratio(n_arms):
    """
    Platform trials share control arm across multiple candidates.
    Efficiency scales roughly 1/sqrt(n_arms) vs. running them separately.
    """
    if n_arms <= 1:
        return 1.0
    return 1.0 / np.sqrt(n_arms)


def recommend_designs(n_candidates, effect_size=DEFAULT_EFFECT_SIZE,
                      alpha=DEFAULT_ALPHA, power=DEFAULT_POWER,
                      cost_per_patient=COST_PER_PATIENT,
                      enrollment_rate=MONTHLY_ENROLLMENT):
    """
    Recommend trial designs for testing n_candidates simultaneously.

    Returns a list of TrialDesign objects, sorted by cost.
    """
    n = max(2, int(n_candidates))
    designs = []

    # --- Fixed: N independent 2-arm trials ---
    n_fixed = _fixed_trial_sample_size(effect_size, alpha, power) * n
    dur_fixed = (n_fixed / enrollment_rate) / n  # parallel enrollment
    cost_fixed = n_fixed * cost_per_patient
    designs.append(TrialDesign(
        name=f"{n} independent 2-arm RCTs",
        n_arms=n * 2,
        expected_sample_size=n_fixed,
        expected_duration_months=round(dur_fixed, 1),
        expected_cost_usd=cost_fixed,
        description="Standard parallel trials. Highest cost, longest duration, most conservative.",
    ))

    # --- Bayesian adaptive multi-arm ---
    n_bayes_base = _fixed_trial_sample_size(effect_size, alpha, power) * n
    n_bayes = int(n_bayes_base * _adaptive_efficiency_ratio(n))
    dur_bayes = n_bayes / enrollment_rate
    cost_bayes = n_bayes * cost_per_patient
    designs.append(TrialDesign(
        name=f"Bayesian adaptive ({n} arms + shared control)",
        n_arms=n + 1,
        expected_sample_size=n_bayes,
        expected_duration_months=round(dur_bayes, 1),
        expected_cost_usd=cost_bayes,
        description="Response-adaptive randomization. Shifts allocation toward effective arms. "
                    "Early stopping for futility. Requires adaptive design expertise.",
    ))

    # --- Platform trial ---
    n_platform_base = _fixed_trial_sample_size(effect_size, alpha, power) * n
    n_platform = int(n_platform_base * _platform_efficiency_ratio(n))
    dur_platform = n_platform / enrollment_rate
    cost_platform = n_platform * cost_per_patient
    designs.append(TrialDesign(
        name=f"Platform trial ({n} arms + control)",
        n_arms=n + 1,
        expected_sample_size=n_platform,
        expected_duration_months=round(dur_platform, 1),
        expected_cost_usd=cost_platform,
        description="Single protocol, multiple arms. Add/drop candidates mid-trial. "
                    "Highest efficiency, requires strong trial infrastructure (e.g., I-SPY, RECOVERY).",
    ))

    designs.sort(key=lambda d: d.expected_cost_usd)
    return designs


def format_recommendation(designs, fixed_cost=None):
    """Format recommendations as a DataFrame."""
    rows = []
    if fixed_cost is None:
        fixed_cost = designs[-1].expected_cost_usd  # assume most expensive = fixed
    for d in designs:
        savings = fixed_cost - d.expected_cost_usd
        savings_pct = (savings / fixed_cost * 100) if fixed_cost > 0 else 0
        rows.append({
            "Design":       d.name,
            "Arms":         d.n_arms,
            "Sample size":  d.expected_sample_size,
            "Duration (mo)": d.expected_duration_months,
            "Cost ($M)":    round(d.expected_cost_usd / 1e6, 1),
            "Savings ($M)": round(savings / 1e6, 1),
            "Savings %":    round(savings_pct, 0),
            "Description":  d.description[:80] + "...",
        })
    return pd.DataFrame(rows)
