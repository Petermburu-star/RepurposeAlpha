"""
RepurposeAlpha — Streamlit app with return estimator + Decision Summary.
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

import auth
import assumptions
import discovery
import correlation
import cache as cache_mod
import returns
import reporting

st.set_page_config(page_title="RepurposeAlpha", page_icon="🧬", layout="wide")
user = auth.require_login(min_role="viewer")

# ---------- Header ----------
col_title, col_user = st.columns([5, 1])
with col_title:
    st.title("🧬 RepurposeAlpha")
with col_user:
    st.caption(f"👤 {user['username']} ({user['role']})")
    if st.button("Log out"):
        auth.log_event(user["username"], "logout")
        st.session_state["user"] = None
        st.rerun()

st.markdown("**Portfolio optimization for drug repurposing.** Type a disease to begin.")
st.divider()

# ---------- Query ----------
st.subheader("🔍 Query")
col_q1, col_q2, col_q3 = st.columns([3, 1, 1])
with col_q1:
    disease_input = st.text_input("Disease name",
        value=st.session_state.get("disease", "Prader-Willi syndrome"))
with col_q2:
    min_phase = st.selectbox("Min phase", [1, 2, 3], index=1)
with col_q3:
    max_candidates = st.number_input("Max candidates", 5, 30, 10, step=5)

col_b1, col_b2, _ = st.columns([1, 1, 4])
with col_b1:
    run_analysis = st.button("🚀 Analyze", type="primary")
with col_b2:
    use_cache = st.checkbox("Use cache", value=True)

if disease_input:
    st.session_state["disease"] = disease_input

if run_analysis and disease_input:
    if use_cache and cache_mod.has_cache(disease_input):
        candidates, corr, meta = cache_mod.load_analysis(disease_input)
        st.success(f"✓ Loaded from cache ({meta.get('n_candidates', '?')} candidates)")
    else:
        with st.status(f"Analyzing '{disease_input}'...", expanded=True) as status:
            st.write("Step 1/3: Discovering candidates...")
            try:
                candidates = discovery.get_candidates_for_disease(
                    disease_input, min_phase=min_phase, max_results=max_candidates)
            except Exception as e:
                status.update(label=f"❌ {e}", state="error"); st.stop()
            if candidates is None or candidates.empty:
                status.update(label="❌ No candidates", state="error"); st.stop()
            st.write(f"   ✓ {len(candidates)} candidates")

            st.write("Step 2/3: Correlation matrix...")
            corr = correlation.build_correlation_matrix(candidates["chembl_id"].tolist(), verbose=False)
            if corr is None or corr.empty:
                status.update(label="❌ Empty matrix", state="error"); st.stop()
            st.write(f"   ✓ {len(corr)}×{len(corr)} matrix")

            st.write("Step 3/3: Saving...")
            cache_mod.save_analysis(disease_input, candidates, corr,
                                     metadata={"min_phase": min_phase})
            status.update(label="✅ Complete", state="complete")

    st.session_state["candidates"] = candidates
    st.session_state["corr"]       = corr
    st.session_state["disease"]    = disease_input

candidates = st.session_state.get("candidates")
corr       = st.session_state.get("corr")
disease    = st.session_state.get("disease", "(none)")

if candidates is None or corr is None:
    st.info("👆 Enter a disease name and click **Analyze** to begin.")
    st.stop()

tickers = list(corr.columns)
drug_names = dict(zip(candidates["chembl_id"], candidates["drug_name"]))

# ---------- Returns (real, not hardcoded) ----------
est = returns.estimate_returns(candidates, corr, verbose=False)
est = est.reindex(tickers).dropna(subset=["mu", "sigma"])

mu_s    = est["mu"]
sigma_s = est["sigma"]
cov = pd.DataFrame(np.outer(sigma_s, sigma_s) * corr.loc[est.index, est.index].values,
                   index=est.index, columns=est.index)

rfr_value = assumptions.get_value("risk_free_rate", 0.03)

# ---------- Sidebar ----------
st.sidebar.header("Assumptions")
st.sidebar.markdown(f"**Risk-free rate:** `{rfr_value:.4f}`")
st.sidebar.caption("Source: FRED DGS10 (live)")
st.sidebar.markdown(f"**PoS source:** BIO/QLS 2021")
st.sidebar.caption("Per-drug returns from rNPV proxy (registry-driven)")

# ---------- Solve ----------
def solve(mode):
    ef = EfficientFrontier(mu_s, cov, weight_bounds=(0, 1))
    try:
        if mode == "sharpe":
            ef.max_sharpe(risk_free_rate=rfr_value)
        else:
            ef.min_volatility()
        w = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=rfr_value)
        return pd.Series(w), ret, vol, sharpe
    except Exception as e:
        return pd.Series(0.0, index=est.index), 0.0, 0.0, 0.0

w_sharpe, r_s, v_s, s_s = solve("sharpe")
w_minvol, r_m, v_m, _   = solve("minvol")

# ============================================================
# DECISION SUMMARY PANEL
# ============================================================
st.markdown(f"### 🎯 Decision summary — {disease.title()}")

# Top candidates by Max Sharpe weight
top = w_sharpe.sort_values(ascending=False)
top4 = top[top > 0.02].head(4)

# Correlated pairs to skip
corr_pairs = []
cids = list(est.index)
for i, a in enumerate(cids):
    for b in cids[i+1:]:
        r = corr.loc[a, b]
        if r > 0.30:
            corr_pairs.append((r, a, b))
corr_pairs.sort(reverse=True)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("**Test these first (Max Sharpe priority):**")
    if len(top4) > 0:
        for cid, w in top4.items():
            name = drug_names.get(cid, cid)
            phase = est.loc[cid, "phase_key"].replace("_", " ")
            mu = est.loc[cid, "mu"]
            st.markdown(f"- **{name}** — weight {w:.3f} · {phase} · est. return {mu:.2f}")
    else:
        st.markdown("_No positive weights — all candidates flagged as high-risk._")

with col2:
    st.markdown("**Skip (redundant):**")
    if corr_pairs:
        seen = set()
        for r, a, b in corr_pairs[:3]:
            key = tuple(sorted([a, b]))
            if key in seen: continue
            seen.add(key)
            st.markdown(f"- {drug_names.get(b,b)} *(r={r:.2f} vs {drug_names.get(a,a)})*")
    else:
        st.markdown("_No redundant pairs detected._")

with col3:
    st.metric("Portfolio Sharpe", f"{s_s:.2f}")
    st.metric("Expected return", f"{r_s:.3f}")
    st.metric("Volatility", f"{v_s:.3f}")

# Flag if returns are degenerate
if est["mu"].std() < 0.02:
    st.warning(
        "⚠️ Return estimates are nearly identical across candidates. "
        "The portfolio optimizer has little signal to exploit. "
        "Consider whether all candidates are at the same clinical stage."
    )

st.divider()

# ============================================================
# TABS
# ============================================================
tab_names = ["🔗 Correlation", "📈 Frontier", "💼 Portfolio",
             "🎚️ Sensitivity", "📋 Assumptions", "📄 Report"]
if auth.has_role(user, "admin"):
    tab_names.append("🛡️ Admin")
tabs = st.tabs(tab_names)

with tabs[0]:
    st.subheader("Candidate correlation matrix")
    labelled = corr.rename(index=drug_names, columns=drug_names)
    fig = px.imshow(labelled, text_auto=".2f", color_continuous_scale="RdYlGn_r",
                    zmin=0, zmax=1, aspect="auto")
    fig.update_layout(height=max(400, 25 * len(tickers)))
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Efficient frontier")
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
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Optimal portfolio weights")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Max Sharpe**")
        df = pd.DataFrame({"weight": w_sharpe})
        df.index = [drug_names.get(i, i) for i in df.index]
        st.dataframe(df.style.format("{:.3f}"))
        st.metric("Expected return", f"{r_s:.3f}")
        st.metric("Volatility",      f"{v_s:.3f}")
        st.metric("Sharpe ratio",    f"{s_s:.3f}")
    with c2:
        st.markdown("**Min Volatility**")
        df = pd.DataFrame({"weight": w_minvol})
        df.index = [drug_names.get(i, i) for i in df.index]
        st.dataframe(df.style.format("{:.3f}"))
        st.metric("Expected return", f"{r_m:.3f}")
        st.metric("Volatility",      f"{v_m:.3f}")

    st.markdown("**Per-drug estimates**")
    disp = est[["drug_name", "phase_key", "mu", "sigma", "peers"]].copy()
    disp.columns = ["Drug", "Phase", "Expected return", "Volatility", "Correlated peers"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Sensitivity analysis")
    if len(tickers) >= 2:
        t1 = st.selectbox("Drug A", tickers, index=0, format_func=lambda x: drug_names.get(x, x))
        t2 = st.selectbox("Drug B", tickers, index=1, format_func=lambda x: drug_names.get(x, x))
        rho = st.slider("Correlation", 0.0, 1.0, float(corr.loc[t1, t2]), 0.01)
        corr2 = corr.copy()
        corr2.loc[t1, t2] = rho; corr2.loc[t2, t1] = rho
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

with tabs[4]:
    st.subheader("📋 Assumption registry")
    reg = assumptions.load_registry()
    rows = []
    def flatten(node, prefix=""):
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict) and "value" in v:
                rows.append({"key": path, "value": v["value"],
                             "confidence": v.get("confidence", "?"),
                             "source": v.get("source", "")})
            elif isinstance(v, dict):
                flatten(v, path)
    flatten(reg)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("📄 Download report")
    st.markdown("Generate a 6-page PDF with methodology, charts, and full provenance.")
    if st.button("📥 Generate PDF report", type="primary"):
        with st.spinner("Generating..."):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            findings = (
                f"Analysis of {disease} with {len(est)} candidates. "
                f"Top allocation: {drug_names.get(top.index[0], top.index[0])} "
                f"at weight {top.iloc[0]:.3f}. "
                f"Portfolio Sharpe ratio: {s_s:.2f}."
            )
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
            st.session_state["pdf_name"]  = f"RepurposeAlpha_{disease.replace(' ', '_')}.pdf"
            tmp_path.unlink()
            st.success(f"✅ Ready ({len(st.session_state['pdf_bytes']):,} bytes)")
    if "pdf_bytes" in st.session_state:
        st.download_button("⬇️  Click to download PDF",
                           data=st.session_state["pdf_bytes"],
                           file_name=st.session_state.get("pdf_name", "report.pdf"),
                           mime="application/pdf", type="primary")

if auth.has_role(user, "admin") and len(tabs) == 7:
    with tabs[6]:
        st.subheader("🛡️ Admin panel")
        st.markdown("**Users**")
        st.dataframe(pd.DataFrame(auth.list_users(), columns=["username", "role", "created_at"]))
        st.markdown("**Audit log**")
        st.dataframe(pd.DataFrame(auth.list_audit(50),
                                  columns=["timestamp", "user", "action", "detail"]))

st.divider()
st.caption(f"RepurposeAlpha · {disease} · {len(est)} candidates · Registry-driven")
