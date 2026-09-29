"""
RepurposeAlpha — Executive Summary engine v2.

Runs all seven features internally, then produces:
- Verdict (RECOMMENDED / CAUTION / NOT RECOMMENDED)
- Plain-English narrative
- Feature scorecard (one line per feature)
- Top candidates with strengths + risks
- Warnings (prioritized)
- Next steps (numbered)

Designed to be the single view a decision-maker reads first.
"""
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Local imports — features must be on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import assumptions
import options
import fto
import trials
import crosstarget
import negativedata
import synergy
import pricing


@dataclass
class FeatureCard:
    icon: str
    title: str
    status: str       # "positive" | "caution" | "negative" | "neutral"
    headline: str     # what the feature found, punchy
    detail: str       # why it matters, plain English


@dataclass
class CandidateVerdict:
    name: str
    phase: str
    weight: float
    expected_return: float
    strengths: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


@dataclass
class Warning:
    level: str
    icon: str
    text: str


@dataclass
class ExecutiveSummary:
    disease: str
    n_candidates: int
    verdict: str
    verdict_color: str
    one_liner: str
    narrative: str
    feature_cards: List[FeatureCard]
    top_candidates: List[CandidateVerdict]
    warnings: List[Warning]
    next_steps: List[str]
    confidence: str


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def estimate_peak_sales(disease_name):
    """Rough peak-sales estimate based on disease category."""
    if not disease_name:
        return 200e6
    d = disease_name.lower()
    if any(k in d for k in ["syndrome", "dystrophy", "deficiency", "gaucher", "fabry", "pompe"]):
        return 500e6
    if any(k in d for k in ["malaria", "tuberculosis", "leishmaniasis", "schistosomiasis",
                             "trypanosomiasis", "chagas", "dengue", "ebola"]):
        return 300e6
    if any(k in d for k in ["cancer", "carcinoma", "leukemia", "lymphoma", "melanoma", "tumor"]):
        return 800e6
    if any(k in d for k in ["diabetes", "hypertension", "cardiovascular", "coronary", "asthma"]):
        return 1000e6
    return 200e6


def estimate_volatility(disease_name):
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


def build_summary(disease, candidates, corr, est, w_sharpe,
                   r_s, v_s, s_s, rfr_value):
    """
    Run all seven features and build the full executive summary.
    """
    drug_names = dict(zip(candidates["chembl_id"], candidates["drug_name"]))
    n = len(candidates)

    # ---------- Run every feature ----------
    fto_df = _safe(fto.assess_portfolio, candidates, verbose=False)

    priors_df = _safe(negativedata.analyze_portfolio, candidates, disease, verbose=False)

    synergy_df = _safe(synergy.find_synergy_pairs, candidates, corr,
                        top_n=3, check_trials=False, verbose=False)

    pricing_df = _safe(pricing.analyze_portfolio, candidates, disease, verbose=False)

    # Real options
    ro_result = None
    try:
        _ph = [
            options.Phase("Phase I", 2.0, float(assumptions.get_value("cost.phase_1_usd", 10e6)),
                          float(assumptions.get_value("pos.phase_1", 0.63))),
            options.Phase("Phase II", 2.0, float(assumptions.get_value("cost.phase_2_usd", 25e6)),
                          float(assumptions.get_value("pos.phase_2", 0.31))),
            options.Phase("Phase III", 3.0, float(assumptions.get_value("cost.phase_3_usd", 100e6)),
                          float(assumptions.get_value("pos.phase_3", 0.58))),
            options.Phase("NDA", 1.0, 5e6, float(assumptions.get_value("pos.nda_bla", 0.85))),
        ]
        ro_result = options.RealOptionsEngine(
            phases=_ph,
            peak_sales=estimate_peak_sales(disease),
            volatility=estimate_volatility(disease),
            risk_free_rate=rfr_value,
            discount_rate=0.10,
        ).value()
    except Exception:
        ro_result = None

    # Trial designs
    trial_designs = _safe(trials.recommend_designs, n_candidates=min(len(est), 6))

    # ---------- Feature scorecard ----------
    cards = []

    # 1. Portfolio
    top_w = w_sharpe.sort_values(ascending=False)
    top_names = [drug_names.get(i, i) for i in top_w.index[:3] if top_w[i] > 0.05]
    if s_s > 1.5:
        p_status, p_headline = "positive", f"Sharpe {s_s:.2f} — strong risk-adjusted return"
    elif s_s > 0.8:
        p_status, p_headline = "positive", f"Sharpe {s_s:.2f} — defensible portfolio"
    elif s_s > 0.4:
        p_status, p_headline = "caution", f"Sharpe {s_s:.2f} — modest signal"
    else:
        p_status, p_headline = "negative", f"Sharpe {s_s:.2f} — weak signal"
    cards.append(FeatureCard(
        "📈", "Portfolio Optimization", p_status, p_headline,
        f"Top picks: {', '.join(top_names) if top_names else 'none above threshold'}",
    ))

    # 2. Real Options
    if ro_result is None:
        cards.append(FeatureCard("⏳", "Real Options", "neutral", "Not available", "—"))
    elif ro_result.option_value > 0:
        cards.append(FeatureCard(
            "⏳", "Real Options", "positive",
            f"PROCEED — option value ${ro_result.option_value:.0f}M",
            f"Flexibility over DCF adds ${ro_result.abandonment_value:.0f}M",
        ))
    else:
        cards.append(FeatureCard(
            "⏳", "Real Options", "negative",
            f"DO NOT START — DCF ${ro_result.asset_value:.0f}M",
            "Even with the option to abandon, this doesn't clear cost of capital.",
        ))

    # 3. FTO
    if fto_df is None or fto_df.empty:
        cards.append(FeatureCard("⚖️", "Freedom-to-Operate", "neutral", "Not available", "—"))
    else:
        n_low = int((fto_df["fto_risk"] == "low").sum())
        n_high = int((fto_df["fto_risk"] == "high").sum())
        if n_high == 0:
            cards.append(FeatureCard(
                "⚖️", "Freedom-to-Operate", "positive",
                f"{n_low}/{n} with strong IP position",
                "No generic-competition blockers.",
            ))
        elif n_high >= n // 2:
            cards.append(FeatureCard(
                "⚖️", "Freedom-to-Operate", "negative",
                f"{n_high}/{n} are generic",
                "Limited commercial protection — no compound patent.",
            ))
        else:
            cards.append(FeatureCard(
                "⚖️", "Freedom-to-Operate", "caution",
                f"{n_low} protected · {n_high} generic",
                "Mixed IP position — review before investment.",
            ))

    # 4. Trials
    if trial_designs:
        cheapest = min(trial_designs, key=lambda d: d.expected_cost_usd)
        baseline = max(trial_designs, key=lambda d: d.expected_cost_usd)
        savings = baseline.expected_cost_usd - cheapest.expected_cost_usd
        cards.append(FeatureCard(
            "🧪", "Trial Design", "positive",
            f"Platform trial saves ${savings/1e6:.1f}M",
            f"vs. {n if n <= 6 else 6} independent RCTs",
        ))
    else:
        cards.append(FeatureCard("🧪", "Trial Design", "neutral", "Not available", "—"))

    # 5. Cross-Disease
    cd_df = _safe(crosstarget.find_cross_disease_overlaps, candidates, disease, verbose=False)
    if cd_df is None or cd_df.empty:
        cards.append(FeatureCard("🌐", "Cross-Disease", "neutral", "Not analyzed", "Click Evidence tab"))
    else:
        n_over = int((cd_df["n_other_diseases"] > 0).sum())
        if n_over >= n * 0.7:
            cards.append(FeatureCard(
                "🌐", "Cross-Disease", "positive",
                f"{n_over}/{n} have other indications",
                "Accumulated safety data — lower regulatory risk.",
            ))
        else:
            cards.append(FeatureCard(
                "🌐", "Cross-Disease", "neutral",
                f"{n_over}/{n} have other indications",
                "Limited prior evidence in other diseases.",
            ))

    # 6. Prior Results
    if priors_df is None or priors_df.empty:
        cards.append(FeatureCard("📉", "Prior Results", "neutral", "Not analyzed", "Click Evidence tab"))
    else:
        n_high = int((priors_df["risk_flag"] == "high").sum())
        n_clean = int((priors_df["risk_flag"] == "clean").sum())
        if n_high == 0:
            cards.append(FeatureCard(
                "📉", "Prior Results", "positive",
                f"{n_clean}/{n} clean — no strong failures",
                "No futility or safety signals in prior trials.",
            ))
        elif n_high >= 2:
            cards.append(FeatureCard(
                "📉", "Prior Results", "negative",
                f"{n_high} with strong prior failures",
                "Futility or safety issues — review before proceeding.",
            ))
        else:
            cards.append(FeatureCard(
                "📉", "Prior Results", "caution",
                f"{n_high} with prior failure signal",
                "At least one candidate has meaningful prior failure history.",
            ))

    # 7. Synergy
    if synergy_df is None or synergy_df.empty:
        cards.append(FeatureCard("🔬", "Synergy", "neutral", "Not analyzed", "Click Evidence tab"))
    else:
        top = synergy_df.iloc[0]
        score = float(top["synergy_score"])
        if score >= 0.55:
            cards.append(FeatureCard(
                "🔬", "Synergy", "positive",
                f"Top pair: {top['drug_a'][:20]} + {top['drug_b'][:20]}",
                f"Synergy score {score:.2f} — good combination candidate.",
            ))
        else:
            cards.append(FeatureCard(
                "🔬", "Synergy", "neutral",
                f"Top pair scored {score:.2f}",
                "No strong synergy signal in this candidate set.",
            ))

    # 8. Pricing
    if pricing_df is None or pricing_df.empty:
        cards.append(FeatureCard("💰", "Pricing", "neutral", "Not available", "—"))
    else:
        n_viable = int((pricing_df["viability"] == "commercially_viable").sum())
        n_not = int((pricing_df["viability"] == "not_viable").sum())
        if n_not == 0:
            cards.append(FeatureCard(
                "💰", "Pricing", "positive",
                f"{n_viable}/{n} commercially viable",
                "Sustainable prices are within market reference range.",
            ))
        elif n_viable >= n // 2:
            cards.append(FeatureCard(
                "💰", "Pricing", "caution",
                f"{n_viable} viable · {n_not} not viable",
                "Mixed commercial position at reference prices.",
            ))
        else:
            cards.append(FeatureCard(
                "💰", "Pricing", "negative",
                f"{n_not}/{n} not commercially viable",
                "Price needed exceeds market reference range.",
            ))

    # ---------- Warnings ----------
    warnings = []
    for card in cards:
        if card.status == "negative":
            warnings.append(Warning("high", card.icon, f"{card.title}: {card.headline}"))
        elif card.status == "caution":
            warnings.append(Warning("medium", card.icon, f"{card.title}: {card.headline}"))

    # ---------- Overall verdict ----------
    n_neg = sum(1 for c in cards if c.status == "negative")
    n_pos = sum(1 for c in cards if c.status == "positive")
    n_cau = sum(1 for c in cards if c.status == "caution")

    if n_neg >= 2:
        verdict, color = "NOT RECOMMENDED", "red"
    elif n_neg == 1 or n_cau >= 3:
        verdict, color = "CAUTION", "yellow"
    elif n_pos >= 4:
        verdict, color = "RECOMMENDED", "green"
    else:
        verdict, color = "REVIEW", "yellow"

    # ---------- Narrative ----------
    narrative_parts = []
    narrative_parts.append(
        f"Analysis of **{disease.title()}** identified **{n} candidate drugs** for repurposing."
    )

    # Portfolio finding
    if top_names:
        narrative_parts.append(
            f"Portfolio optimization ranks **{top_names[0]}** first "
            f"(weight {top_w[top_w.index[0]]:.2f}), with a portfolio Sharpe ratio of **{s_s:.2f}**."
        )

    # Real options finding
    if ro_result is not None:
        if ro_result.option_value > 0:
            narrative_parts.append(
                f"Real options analysis says **PROCEED** — the flexibility to abandon "
                f"is worth ${ro_result.abandonment_value:.0f}M over a naive DCF."
            )
        else:
            narrative_parts.append(
                f"Real options analysis says **DO NOT START** from scratch — the DCF is "
                f"${ro_result.asset_value:.0f}M. The option value is zero."
            )

    # FTO finding
    if fto_df is not None and not fto_df.empty:
        n_high = int((fto_df["fto_risk"] == "high").sum())
        if n_high > 0:
            narrative_parts.append(
                f"On commercial protection, **{n_high} of {n} candidates are generic** — no compound patent."
            )
        else:
            narrative_parts.append(
                f"All candidates have **at least partial IP protection** — a good sign for commercial viability."
            )

    # Prior results
    if priors_df is not None and not priors_df.empty:
        n_high_p = int((priors_df["risk_flag"] == "high").sum())
        if n_high_p > 0:
            narrative_parts.append(
                f"Prior trial history shows **{n_high_p} candidate(s) with meaningful failure signals**."
            )

    # Trials
    if trial_designs:
        cheapest = min(trial_designs, key=lambda d: d.expected_cost_usd)
        baseline = max(trial_designs, key=lambda d: d.expected_cost_usd)
        savings = baseline.expected_cost_usd - cheapest.expected_cost_usd
        narrative_parts.append(
            f"If you proceed to validation, using a **{cheapest.name.split('(')[0].strip().lower()}** "
            f"would save **${savings/1e6:.1f}M** vs. independent trials."
        )

    narrative_parts.append(f"**Overall verdict: {verdict}.**")
    narrative = " ".join(narrative_parts)

    # ---------- One-liner ----------
    if verdict == "RECOMMENDED":
        one_liner = "The portfolio is well-positioned. Proceed with validation planning."
    elif verdict == "CAUTION":
        one_liner = "Proceed with caution — resolve the flagged issues first."
    elif verdict == "NOT RECOMMENDED":
        one_liner = "Multiple critical issues. Reassess before committing resources."
    else:
        one_liner = "Review the feature scorecard for details."

    # ---------- Top candidates ----------
    top_ids = [i for i in top_w.index if top_w[i] > 0.02][:5]
    top_candidates = []
    for cid in top_ids:
        name = drug_names.get(cid, cid)
        weight = float(top_w[cid])
        phase_key = est.loc[cid, "phase_key"] if cid in est.index else "unknown"
        phase = phase_key.replace("_", " ").title()
        mu = float(est.loc[cid, "mu"]) if cid in est.index else 0.0

        strengths = []
        risks = []

        if weight > 0.15:
            strengths.append(f"Highest portfolio priority (weight {weight:.2f})")

        # FTO
        if fto_df is not None and not fto_df.empty:
            row = fto_df[fto_df["chembl_id"] == cid]
            if not row.empty:
                score = float(row.iloc[0]["fto_score"])
                if score >= 0.65:
                    strengths.append("Strong IP position")
                elif score < 0.30:
                    risks.append("Generic — no commercial protection")

        # Correlation
        correlated = [(drug_names.get(j, j), float(corr.loc[cid, j]))
                      for j in corr.columns if j != cid and corr.loc[cid, j] > 0.4]
        if correlated:
            peer, r = max(correlated, key=lambda x: x[1])
            risks.append(f"Correlated with {peer} (r={r:.2f}) — potentially redundant")

        # Prior results
        if priors_df is not None and not priors_df.empty:
            row = priors_df[priors_df["chembl_id"] == cid]
            if not row.empty and row.iloc[0]["risk_flag"] in ("high", "medium"):
                risks.append(f"Prior failure signal ({row.iloc[0]['risk_flag']})")

        top_candidates.append(CandidateVerdict(
            name=name, phase=phase, weight=weight,
            expected_return=mu, strengths=strengths, risks=risks,
        ))

    # ---------- Next steps ----------
    next_steps = []
    if top_names:
        next_steps.append(
            f"Prioritize validation budget on **{', '.join(top_names[:3])}**"
        )
    if priors_df is not None and int((priors_df["risk_flag"] == "high").sum()) > 0:
        next_steps.append(
            "Review prior failure details for flagged candidates before committing"
        )
    if fto_df is not None and int((fto_df["fto_risk"] == "high").sum()) > 0:
        next_steps.append(
            "Define IP strategy for generic candidates (formulation, method-of-use)"
        )
    if trial_designs:
        cheapest = min(trial_designs, key=lambda d: d.expected_cost_usd)
        next_steps.append(
            f"Plan a **{cheapest.name.split('(')[0].strip().lower()}** — reduces cost by "
            f"${(max(d.expected_cost_usd for d in trial_designs) - cheapest.expected_cost_usd)/1e6:.1f}M"
        )
    if synergy_df is not None and not synergy_df.empty:
        top_pair = synergy_df.iloc[0]
        next_steps.append(
            f"Consider combination testing: **{top_pair['drug_a']} + {top_pair['drug_b']}**"
        )
    if not next_steps:
        next_steps.append("Review each tab for detailed analysis")

    # ---------- Confidence ----------
    available = sum(1 for c in cards if c.status != "neutral")
    if available >= 7:
        confidence = "high"
    elif available >= 5:
        confidence = "medium"
    else:
        confidence = "low"

    return ExecutiveSummary(
        disease=disease,
        n_candidates=n,
        verdict=verdict,
        verdict_color=color,
        one_liner=one_liner,
        narrative=narrative,
        feature_cards=cards,
        top_candidates=top_candidates,
        warnings=warnings,
        next_steps=next_steps,
        confidence=confidence,
    )
