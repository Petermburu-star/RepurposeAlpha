"""
RepurposeAlpha — Streamlit app v0.3.0 (organized).
Seven top-level tabs. Executive Summary first.
"""
import sys, io
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT    = APP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import EfficientFrontier

import auth, assumptions, discovery, correlation, cache as cache_mod
import returns, reporting, options, fto, trials, crosstarget
import negativedata, synergy, pricing, executive

st.set_page_config(page_title="RepurposeAlpha", page_icon="🧬", layout="wide")

# ---------- Auth ----------
user = auth.require_login(min_role="viewer")

# ---------- Header ----------
col_title, col_user = st.columns([5, 1])
with col_title:
    st.title("🧬 RepurposeAlpha")
    st.caption("Portfolio decision support for drug repurposing")
with col_user:
    st.caption(f"👤 {user['username']} ({user['role']})")
    if st.button("Log out"):
        auth.log_event(user["username"], "logout")
        st.session_state["user"] = None
        st.rerun()

# ============================================================
# QUERY BAR
# ============================================================
with st.container():
    q1, q2, q3, q4, q5 = st.columns([3, 1, 1, 1, 1])
    with q1:
        disease_input = st.text_input("Disease", value=st.session_state.get("disease", "Prader-Willi syndrome"))
    with q2:
        min_phase = st.selectbox("Min phase", [1, 2, 3], index=1)
    with q3:
        max_candidates = st.number_input("Max", 5, 30, 10, step=5)
    with q4:
        use_cache = st.checkbox("Cache", value=True)
    with q5:
        st.write("")
        run_analysis = st.button("🚀 Analyze", type="primary")

if disease_input:
    st.session_state["disease"] = disease_input

if run_analysis and disease_input:
    if use_cache and cache_mod.has_cache(disease_input, min_phase=min_phase, max_candidates=max_candidates):
        candidates, corr, meta = cache_mod.load_analysis(disease_input, min_phase=min_phase, max_candidates=max_candidates)
        st.success(f"✓ Loaded from cache ({meta.get('n_candidates', '?')} candidates)")
    else:
        with st.status(f"Analyzing {disease_input}...", expanded=True) as status:
            st.write("1/3 Discovering candidates from Open Targets...")
            try:
                candidates = discovery.get_candidates_for_disease(disease_input, min_phase=min_phase, max_results=max_candidates)
            except Exception as e:
                status.update(label=f"❌ Discovery failed: {e}", state="error"); st.stop()
            if candidates is None or candidates.empty:
                status.update(label=f"❌ No candidates found", state="error"); st.stop()
            st.write(f"   ✓ {len(candidates)} candidates")

            st.write("2/3 Computing correlation matrix...")
            try:
                corr = correlation.build_correlation_matrix(candidates["chembl_id"].tolist(), verbose=False)
            except Exception as e:
                status.update(label=f"❌ Correlation failed: {e}", state="error"); st.stop()
            if corr is None or corr.empty:
                status.update(label="❌ Empty matrix", state="error"); st.stop()
            st.write(f"   ✓ {len(corr)}×{len(corr)} matrix")

            st.write("3/3 Caching...")
            cache_mod.save_analysis(disease_input, candidates, corr,
                                     min_phase=min_phase, max_candidates=max_candidates,
                                     metadata={"min_phase": min_phase, "max_candidates": max_candidates})
            status.update(label=f"✅ Complete", state="complete")

    st.session_state["candidates"] = candidates
    st.session_state["corr"] = corr
    st.session_state["disease"] = disease_input
    # Reset downstream caches on new analysis
    for key in ["fto_df", "nd_df", "syn_df", "pr_df"]:
        st.session_state.pop(key, None)

candidates = st.session_state.get("candidates")
corr       = st.session_state.get("corr")
disease    = st.session_state.get("disease", "(none)")

if candidates is None or corr is None:
    st.info("👆 Enter a disease name and click **Analyze** to begin.")
    st.stop()

# ---------- Compute base analyses ----------
tickers = list(corr.columns)
drug_names = dict(zip(candidates["chembl_id"], candidates["drug_name"]))

est = returns.estimate_returns(candidates, corr, verbose=False)
est = est.reindex(tickers).dropna(subset=["mu", "sigma"])
mu_s = est["mu"]
sigma_s = est["sigma"]

rfr_entry = assumptions.load_registry().get("risk_free_rate", {})
rfr_value = float(rfr_entry.get("value", 0.03))

cov = pd.DataFrame(np.outer(sigma_s, sigma_s) * corr.loc[est.index, est.index].values,
                   index=est.index, columns=est.index)

def solve(mode):
    ef = EfficientFrontier(mu_s, cov, weight_bounds=(0, 1))
    try:
        if mode == "sharpe": ef.max_sharpe(risk_free_rate=rfr_value)
        else: ef.min_volatility()
        w = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=rfr_value)
        return pd.Series(w), ret, vol, sharpe
    except Exception:
        return pd.Series(0.0, index=est.index), 0.0, 0.0, 0.0

w_sharpe, r_s, v_s, s_s = solve("sharpe")
w_minvol, r_m, v_m, _   = solve("minvol")

# ---------- Sidebar ----------
st.sidebar.header("Context")
st.sidebar.markdown(f"**Disease:** {disease}")
st.sidebar.markdown(f"**Candidates:** {len(est)}")
st.sidebar.markdown(f"**Risk-free rate:** {rfr_value:.4f}")
st.sidebar.caption("Source: FRED DGS10 (live)")

# ============================================================
# MAIN TABS (7 top-level)
# ============================================================
tab_names = ["📊 Executive Summary", "📈 Portfolio", "🧬 Strategic",
             "📚 Evidence", "⚙️ Assumptions", "📄 Report"]
if auth.has_role(user, "admin"):
    tab_names.append("🛡️ Admin")
tabs = st.tabs(tab_names)


# ---------- TAB 1: EXECUTIVE SUMMARY ----------
with tabs[0]:
    with st.spinner("Running all seven features..."):
        summary = executive.build_summary(
            disease=disease, candidates=candidates, corr=corr,
            est=est, w_sharpe=w_sharpe,
            r_s=r_s, v_s=v_s, s_s=s_s, rfr_value=rfr_value,
        )

    # ---- VERDICT BANNER ----
    color_map = {"green": "#10b981", "yellow": "#f59e0b", "red": "#ef4444"}
    vc = color_map.get(summary.verdict_color, "#64748b")
    st.markdown(
        f"""<div style="background:{vc}22; border-left:5px solid {vc};
        padding:22px 26px; border-radius:8px; margin-bottom:24px;">
        <div style="font-size:11px; color:#94a3b8; letter-spacing:1.8px; text-transform:uppercase;">Overall verdict</div>
        <div style="font-size:34px; font-weight:700; color:{vc}; margin:6px 0 10px 0;">{summary.verdict}</div>
        <div style="font-size:15px; color:#cbd5e1;">{summary.one_liner}</div>
        </div>""", unsafe_allow_html=True)

    st.caption(
        "Verdict considers all seven features. **Real Options is a hard veto** — "
        "if the project doesn't clear cost of capital, verdict cannot be RECOMMENDED. "
        "Adjust assumptions in the Strategic tab to see how the verdict changes."
    )

    # ---- NARRATIVE ----
    st.markdown("### 📖 Summary")
    st.markdown(f"<div style='font-size:15px; line-height:1.75; color:#cbd5e1;'>{summary.narrative}</div>", unsafe_allow_html=True)

    # ---- FEATURE SCORECARD ----
    st.markdown("### 🎯 Feature Scorecard")
    st.caption("What each of the seven features found, in one line.")
    status_colors = {
        "positive": ("#10b981", "✅"),
        "caution":  ("#f59e0b", "⚠️"),
        "negative": ("#ef4444", "❌"),
        "neutral":  ("#64748b", "○"),
    }
    cols = st.columns(2)
    for i, card in enumerate(summary.feature_cards):
        with cols[i % 2]:
            color, badge = status_colors[card.status]
            st.markdown(
                f"""<div style="background:#1e293b; border-left:4px solid {color};
                padding:14px 18px; border-radius:6px; margin-bottom:12px;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                    <span style="font-size:18px;">{card.icon}</span>
                    <span style="font-weight:600; color:#f1f5f9;">{card.title}</span>
                    <span style="margin-left:auto; color:{color};">{badge}</span>
                </div>
                <div style="color:{color}; font-size:14px; font-weight:500; margin-bottom:4px;">{card.headline}</div>
                <div style="color:#94a3b8; font-size:13px;">{card.detail}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()

    # ---- TOP CANDIDATES ----
    st.markdown("### 🏆 Top candidates")
    if summary.top_candidates:
        for i, c in enumerate(summary.top_candidates[:5], 1):
            with st.expander(f"**{i}. {c.name}** · {c.phase} · weight {c.weight:.3f}", expanded=(i <= 2)):
                c1_, c2_ = st.columns(2)
                with c1_:
                    st.markdown("**Strengths**")
                    if c.strengths:
                        for s in c.strengths:
                            st.markdown(f"✅ {s}")
                    else:
                        st.caption("No specific strengths flagged")
                with c2_:
                    st.markdown("**Risks**")
                    if c.risks:
                        for r in c.risks:
                            st.markdown(f"⚠️ {r}")
                    else:
                        st.caption("No specific risks flagged")
                st.caption(f"Expected annual return: {c.expected_return:.3f}")
    else:
        st.info("No candidates passed the portfolio threshold.")

    st.divider()

    # ---- WARNINGS ----
    if summary.warnings:
        st.markdown("### ⚠️ Items to review")
        for w in summary.warnings:
            if w.level == "high":
                st.warning(f"{w.icon} {w.text}")
            else:
                st.info(f"{w.icon} {w.text}")

    # ---- NEXT STEPS ----
    st.markdown("### 📋 Recommended next steps")
    for i, step in enumerate(summary.next_steps, 1):
        st.markdown(f"**{i}.** {step}")

    st.divider()
    st.caption(f"Confidence: **{summary.confidence}** · Full details in the tabs below · PDF in Report tab")


# ---------- TAB 2: PORTFOLIO ----------
with tabs[1]:
    sub = st.tabs(["🔗 Correlation", "📈 Frontier", "💼 Weights", "🎚️ Sensitivity"])

    with sub[0]:
        st.markdown("**Biological similarity between candidates.** 1.0 = same bet, 0.0 = independent.")
        labelled = corr.rename(index=drug_names, columns=drug_names)
        fig = px.imshow(labelled, text_auto=".2f", color_continuous_scale="RdYlGn_r", zmin=0, zmax=1, aspect="auto")
        fig.update_layout(height=max(400, 25 * len(tickers)))
        st.plotly_chart(fig, width="stretch")

    with sub[1]:
        st.markdown("**Risk-return tradeoff.** Each point is a random portfolio.")
        n = 3000
        rng = np.random.default_rng(42)
        rand_w = rng.dirichlet(np.ones(len(mu_s)), n)
        rets = rand_w @ mu_s.values
        vols = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, cov.values, rand_w))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vols, y=rets, mode="markers",
                                 marker=dict(size=4, color=rets/vols, colorscale="Viridis")))
        fig.add_trace(go.Scatter(x=[v_s, v_m], y=[r_s, r_m], mode="markers+text",
                                 marker=dict(size=18, color=["gold", "silver"]),
                                 text=["Max Sharpe", "Min Vol"], textposition="top center"))
        fig.update_layout(xaxis_title="Risk", yaxis_title="Expected return", height=520)
        st.plotly_chart(fig, width="stretch")

    with sub[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Max Sharpe**")
            df = pd.DataFrame({"weight": w_sharpe})
            df.index = [drug_names.get(i, i) for i in df.index]
            st.dataframe(df.style.format("{:.3f}"), width="stretch")
            st.metric("Return", f"{r_s:.3f}")
            st.metric("Volatility", f"{v_s:.3f}")
            st.metric("Sharpe", f"{s_s:.3f}")
        with c2:
            st.markdown("**Min Volatility**")
            df = pd.DataFrame({"weight": w_minvol})
            df.index = [drug_names.get(i, i) for i in df.index]
            st.dataframe(df.style.format("{:.3f}"), width="stretch")
            st.metric("Return", f"{r_m:.3f}")
            st.metric("Volatility", f"{v_m:.3f}")

    with sub[3]:
        if len(tickers) >= 2:
            t1 = st.selectbox("Drug A", tickers, index=0, format_func=lambda x: drug_names.get(x, x))
            t2 = st.selectbox("Drug B", tickers, index=1, format_func=lambda x: drug_names.get(x, x))
            rho = st.slider("Correlation", 0.0, 1.0, float(corr.loc[t1, t2]), 0.01)
            corr2 = corr.copy(); corr2.loc[t1, t2] = rho; corr2.loc[t2, t1] = rho
            cov2 = pd.DataFrame(np.outer(sigma_s, sigma_s) * corr2.loc[est.index, est.index].values,
                                index=est.index, columns=est.index)
            ef = EfficientFrontier(mu_s, cov2, weight_bounds=(0, 1))
            try:
                ef.max_sharpe(risk_free_rate=rfr_value)
                w = pd.Series(ef.clean_weights())
                w.index = [drug_names.get(i, i) for i in w.index]
                st.bar_chart(w)
            except Exception as e:
                st.error(f"Solver error: {e}")


# ---------- TAB 3: STRATEGIC ----------
with tabs[2]:
    sub = st.tabs(["⏳ Real Options", "⚖️ FTO", "🧪 Trials", "💰 Pricing"])

    with sub[0]:
        st.markdown("**When should you kill this project?** Each candidate is modeled as a series of phase gates.")
        c1_, c2_ = st.columns([1, 2])
        with c1_:
            _default_peak = int(executive.estimate_peak_sales(disease) / 1e6)
            _default_vol  = executive.estimate_volatility(disease)
            peak = st.number_input("Peak sales ($M)", 50, 2000, _default_peak, step=50) * 1e6
            vol  = st.slider("Volatility", 0.20, 0.80, _default_vol, 0.05)
            st.caption(f"Estimated from disease category: {disease}")
        _ph = [
            options.Phase("Phase I", 2.0, float(assumptions.get_value("cost.phase_1_usd", 10e6)), float(assumptions.get_value("pos.phase_1", 0.63))),
            options.Phase("Phase II", 2.0, float(assumptions.get_value("cost.phase_2_usd", 25e6)), float(assumptions.get_value("pos.phase_2", 0.31))),
            options.Phase("Phase III", 3.0, float(assumptions.get_value("cost.phase_3_usd", 100e6)), float(assumptions.get_value("pos.phase_3", 0.58))),
            options.Phase("NDA", 1.0, 5e6, float(assumptions.get_value("pos.nda_bla", 0.85))),
        ]
        r = options.RealOptionsEngine(phases=_ph, peak_sales=peak, volatility=vol,
                                      risk_free_rate=rfr_value, discount_rate=0.10).value()
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("DCF", f"${r.asset_value:.1f}M")
        mc2.metric("Option value", f"${r.option_value:.1f}M")
        mc3.metric("Flexibility premium", f"${r.abandonment_value:.1f}M")
        if r.option_value > 0:
            st.success(f"✅ PROCEED — option value ${r.option_value:.1f}M")
        else:
            st.error(f"❌ DO NOT START — DCF ${r.asset_value:.1f}M")
        gate_df = pd.DataFrame(r.decision_tree)
        st.dataframe(gate_df, width="stretch", hide_index=True)

    with sub[1]:
        st.markdown("**Commercial IP position.** Screening only — not legal advice.")
        fto_tab = fto.assess_portfolio(candidates, verbose=False)
        mc1, mc2, mc3 = st.columns(3)
        counts = fto_tab["fto_risk"].value_counts().to_dict()
        mc1.metric("🟢 Low", counts.get("low", 0))
        mc2.metric("🟡 Medium", counts.get("medium", 0))
        mc3.metric("🔴 High", counts.get("high", 0))
        disp = fto_tab[[
            "drug_name", "status", "fto_risk", "n_patents",
            "latest_patent", "n_excl", "source", "rationale"
        ]].copy()
        disp.columns = ["Drug", "Stage", "Risk", "# Patents", "Latest expiry", "# Excl", "Source", "Rationale"]
        st.dataframe(disp, width="stretch", hide_index=True)

        st.caption(
            "**Source:** 'openfda' = real FDA Orange Book data · "
            "'heuristic' = drug not in FDA data, screening estimate only. "
            "This is a screening signal, not legal advice."
        )

    with sub[2]:
        st.markdown("**Trial architecture.** Platform trials share control arms — usually cheaper and faster.")
        n_test = st.slider("Candidates to test", 2, 10, min(len(est), 6))
        designs = trials.recommend_designs(n_candidates=n_test)
        df_t = trials.format_recommendation(designs)
        cheapest = min(designs, key=lambda d: d.expected_cost_usd)
        baseline = max(designs, key=lambda d: d.expected_cost_usd)
        savings = baseline.expected_cost_usd - cheapest.expected_cost_usd
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Cheapest", cheapest.name.split("(")[0].strip(), f"${cheapest.expected_cost_usd/1e6:.1f}M")
        mc2.metric("Fastest", min(designs, key=lambda d: d.expected_duration_months).name.split("(")[0].strip())
        mc3.metric("Savings vs fixed", f"${savings/1e6:.1f}M")
        st.dataframe(df_t[["Design", "Arms", "Sample size", "Duration (mo)", "Cost ($M)", "Savings %"]],
                     width="stretch", hide_index=True)

    with sub[3]:
        st.markdown("**Sustainable price per patient per year.** Recovers development cost + WACC.")
        pr_df = pricing.analyze_portfolio(candidates, disease, verbose=False)
        counts = pr_df["viability"].value_counts().to_dict()
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("🟢 Viable", counts.get("commercially_viable", 0))
        mc2.metric("🟡 Marginal", counts.get("marginal", 0))
        mc3.metric("🔴 Not viable", counts.get("not_viable", 0))
        disp = pr_df[["drug_name", "phase", "sustainable_usd", "reference_usd", "viability"]].copy()
        disp.columns = ["Drug", "Phase", "Sustainable $/yr", "Reference $/yr", "Viability"]
        st.dataframe(disp, width="stretch", hide_index=True)


# ---------- TAB 4: EVIDENCE ----------
with tabs[3]:
    sub = st.tabs(["🌐 Cross-Disease", "📉 Prior Results", "🔬 Synergy"])

    with sub[0]:
        st.markdown("**Other diseases where each candidate is tested.** More indications = more accumulated safety data.")
        if st.button("Analyze cross-disease", key="cd_btn"):
            with st.spinner("Querying Open Targets..."):
                st.session_state["cd_result"] = crosstarget.find_cross_disease_overlaps(candidates, disease, verbose=False)
        cd_df = st.session_state.get("cd_result")
        if cd_df is not None and not cd_df.empty:
            n_over = int((cd_df["n_other_diseases"] > 0).sum())
            st.metric("Cross-disease hits", f"{n_over}/{len(cd_df)}")
            disp = cd_df[["drug_name", "n_other_diseases", "top_other_disease", "all_other_diseases"]].copy()
            disp.columns = ["Drug", "# Others", "Top", "All overlaps"]
            st.dataframe(disp, width="stretch", hide_index=True)
        else:
            st.info("Click above to run cross-disease analysis.")

    with sub[1]:
        st.markdown("**Prior terminated trials.** Weighted by failure reason (futility/safety = strong; enrollment/business = weak).")
        if st.button("Analyze prior results", key="nd_btn"):
            with st.spinner("Querying ClinicalTrials.gov..."):
                st.session_state["nd_result"] = negativedata.analyze_portfolio(candidates, disease, verbose=False)
        nd_df = st.session_state.get("nd_result")
        if nd_df is not None and not nd_df.empty:
            counts = nd_df["risk_flag"].value_counts().to_dict()
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("🔴 High", counts.get("high", 0))
            mc2.metric("🟡 Medium", counts.get("medium", 0))
            mc3.metric("🟢 Low", counts.get("low", 0))
            mc4.metric("✅ Clean", counts.get("clean", 0))
            disp = nd_df[["drug_name", "n_failed", "n_relevant", "risk_flag", "weighted_risk", "breakdown"]].copy()
            disp.columns = ["Drug", "# Failed", "# Relevant", "Risk", "Weighted", "Breakdown"]
            st.dataframe(disp, width="stretch", hide_index=True)
        else:
            st.info("Click above to check prior failures.")

    with sub[2]:
        st.markdown("**Pairs to test together.** Peak synergy at correlation 0.1–0.5 — shared context, different mechanism.")
        if st.button("Analyze synergy", key="syn_btn"):
            with st.spinner("Scoring pairs..."):
                st.session_state["syn_result"] = synergy.find_synergy_pairs(candidates, corr, top_n=15, check_trials=False, verbose=False)
        syn_df = st.session_state.get("syn_result")
        if syn_df is not None and not syn_df.empty:
            top = synergy.top_combination_recommendation(syn_df)
            if top.get("found"):
                st.success(f"🎯 Top pair: **{top['drug_a']} + {top['drug_b']}** (score {top['score']:.2f})")

            # Coverage note
            n_real = int((syn_df["source"] == "synergxdb").sum()) if "source" in syn_df.columns else 0
            if n_real == 0:
                st.caption(
                    "ℹ️ **No SYNERGxDB data for these pairs.** SYNERGxDB covers small-molecule cancer "
                    "combinations only. For antibodies and non-cancer drugs, we use a correlation-based "
                    "heuristic — scores are estimates, not experimental results."
                )
            else:
                st.caption(f"✓ {n_real}/{len(syn_df)} pairs have real SYNERGxDB data.")
            disp = syn_df[["drug_a", "drug_b", "correlation", "synergy_score", "rationale"]].copy()
            disp.columns = ["Drug A", "Drug B", "Correlation", "Synergy", "Rationale"]
            st.dataframe(disp, width="stretch", hide_index=True)
        else:
            st.info("Click above to score combinations.")


# ---------- TAB 5: ASSUMPTIONS ----------
with tabs[4]:
    st.subheader("Assumption registry")
    st.caption("Every numeric input with source, confidence, and timestamp.")
    reg = assumptions.load_registry()
    rows = []
    def _flat(node, prefix=""):
        for k, v in node.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict) and "value" in v:
                rows.append({"Key": p, "Value": v["value"], "Confidence": v.get("confidence", "?"),
                             "Source": v.get("source", "")})
            elif isinstance(v, dict):
                _flat(v, p)
    _flat(reg)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---------- TAB 6: REPORT ----------
with tabs[5]:
    st.subheader("📄 Download report")
    st.markdown("Generate a 6-page PDF with methodology, charts, and full provenance.")
    if st.button("📥 Generate PDF report", type="primary", key="pdf_btn"):
        with st.spinner("Generating..."):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            findings = f"Analysis of {disease} with {len(est)} candidates. Portfolio Sharpe: {s_s:.2f}."
            reporting.build_report(
                path=tmp_path, title=f"RepurposeAlpha — {disease.title()}",
                user=user["username"], disease=disease,
                corr=corr.rename(index=drug_names, columns=drug_names),
                mu=mu_s, sigma=sigma_s, cov=cov,
                w_sharpe=w_sharpe.rename(index=drug_names),
                w_minvol=w_minvol.rename(index=drug_names),
                registry=assumptions.load_registry(), findings=findings,
            )
            st.session_state["pdf_bytes"] = tmp_path.read_bytes()
            st.session_state["pdf_name"] = f"RepurposeAlpha_{disease.replace(' ', '_')}.pdf"
            tmp_path.unlink()
            st.success(f"✅ Ready ({len(st.session_state['pdf_bytes']):,} bytes)")
    if "pdf_bytes" in st.session_state:
        st.download_button("⬇️  Download PDF", data=st.session_state["pdf_bytes"],
                           file_name=st.session_state.get("pdf_name", "report.pdf"),
                           mime="application/pdf", type="primary")


# ---------- TAB 7: ADMIN ----------
if auth.has_role(user, "admin") and len(tabs) == 7:
    with tabs[6]:
        st.subheader("🛡️ Admin panel")
        st.markdown("**Users**")
        st.dataframe(pd.DataFrame(auth.list_users(), columns=["username", "role", "created_at"]),
                     width="stretch", hide_index=True)
        st.markdown("**Audit log**")
        st.dataframe(pd.DataFrame(auth.list_audit(50),
                                  columns=["timestamp", "user", "action", "detail"]),
                     width="stretch", hide_index=True)


st.divider()
st.caption(f"RepurposeAlpha v0.3.0 · {disease} · {len(est)} candidates")
