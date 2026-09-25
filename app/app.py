"""
RepurposeAlpha — Streamlit app with disease query interface.
"""
import sys
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

st.set_page_config(page_title="RepurposeAlpha", page_icon="🧬", layout="wide")

user = auth.require_login(min_role="viewer")

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

# ============================================================
# QUERY INTERFACE
# ============================================================
st.subheader("🔍 Query")

col_q1, col_q2, col_q3 = st.columns([3, 1, 1])
with col_q1:
    disease_input = st.text_input(
        "Disease name",
        value=st.session_state.get("disease", "Prader-Willi syndrome"),
        placeholder="e.g. tuberculosis, Alzheimer disease, type 2 diabetes",
    )
with col_q2:
    min_phase = st.selectbox("Min phase", [1, 2, 3], index=1)
with col_q3:
    max_candidates = st.number_input("Max candidates", 5, 30, 10, step=5)

col_b1, col_b2, col_b3 = st.columns([1, 1, 4])
with col_b1:
    run_analysis = st.button("🚀 Analyze", type="primary")
with col_b2:
    use_cache = st.checkbox("Use cache", value=True)

if disease_input:
    st.session_state["disease"] = disease_input

if run_analysis and disease_input:
    candidates = None
    corr = None
    meta = None

    if use_cache and cache_mod.has_cache(disease_input):
        st.info(f"📦 Loading cached analysis for '{disease_input}'...")
        candidates, corr, meta = cache_mod.load_analysis(disease_input)
        st.success(f"✓ Loaded from cache ({meta.get('n_candidates', '?')} candidates)")
    else:
        with st.status(f"Analyzing '{disease_input}'...", expanded=True) as status:
            st.write("Step 1/3: Discovering candidate drugs from Open Targets...")
            try:
                candidates = discovery.get_candidates_for_disease(
                    disease_input, min_phase=min_phase, max_results=max_candidates
                )
            except Exception as e:
                status.update(label=f"❌ Discovery failed: {e}", state="error")
                st.stop()

            if candidates is None or candidates.empty:
                status.update(label=f"❌ No candidates found for '{disease_input}'", state="error")
                st.stop()

            st.write(f"   ✓ Found {len(candidates)} candidates")

            st.write("Step 2/3: Computing correlation matrix...")
            try:
                ids = candidates["chembl_id"].tolist()
                corr = correlation.build_correlation_matrix(ids, verbose=False)
            except Exception as e:
                status.update(label=f"❌ Correlation failed: {e}", state="error")
                st.stop()

            if corr is None or corr.empty:
                status.update(label="❌ Correlation matrix empty", state="error")
                st.stop()

            st.write(f"   ✓ Correlation matrix built ({len(corr)} × {len(corr)})")

            st.write("Step 3/3: Saving to cache...")
            cache_mod.save_analysis(disease_input, candidates, corr,
                                     metadata={"min_phase": min_phase})
            status.update(label=f"✅ Analysis complete for '{disease_input}'", state="complete")

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

# ---------- Sidebar ----------
st.sidebar.header("Assumptions")
rfr_entry = assumptions.load_registry().get("risk_free_rate", {})
rfr_value  = rfr_entry.get("value", 0.02)
st.sidebar.markdown(f"**Risk-free rate:** `{rfr_value:.4f}`")
st.sidebar.caption(f"Source: {rfr_entry.get('source', 'unknown')}")

pos_phase3 = assumptions.get_value("pos.phase_3", 0.58)
st.sidebar.caption(f"PoS Phase 3 baseline: {pos_phase3:.2f} (BIO/QLS)")

mu_s    = pd.Series({t: pos_phase3 for t in tickers})
sigma_s = pd.Series({t: 0.30 for t in tickers})
cov = pd.DataFrame(np.outer(sigma_s, sigma_s) * corr.values, index=tickers, columns=tickers)

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
        st.warning(f"Solver failed: {e}")
        return pd.Series(0.0, index=tickers), 0.0, 0.0, 0.0

w_sharpe, r_s, v_s, s_s = solve("sharpe")
w_minvol, r_m, v_m, _   = solve("minvol")

st.markdown(f"### Analysis: **{disease}**  ·  {len(tickers)} candidates")

tab_names = ["🔗 Correlation", "📈 Frontier", "💼 Portfolio", "🎚️ Sensitivity", "📋 Assumptions"]
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
    rand_w = rng.dirichlet(np.ones(len(tickers)), n)
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

with tabs[3]:
    st.subheader("Sensitivity analysis")
    if len(tickers) >= 2:
        t1 = st.selectbox("Drug A", tickers, index=0, format_func=lambda x: drug_names.get(x, x))
        t2 = st.selectbox("Drug B", tickers, index=1, format_func=lambda x: drug_names.get(x, x))
        rho = st.slider("Correlation", 0.0, 1.0, float(corr.loc[t1, t2]), 0.01)
        corr2 = corr.copy()
        corr2.loc[t1, t2] = rho
        corr2.loc[t2, t1] = rho
        cov2 = pd.DataFrame(np.outer(sigma_s, sigma_s) * corr2.values, index=tickers, columns=tickers)
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

if auth.has_role(user, "admin") and len(tabs) == 6:
    with tabs[5]:
        st.subheader("🛡️ Admin panel")
        st.markdown("**Users**")
        st.dataframe(pd.DataFrame(auth.list_users(), columns=["username", "role", "created_at"]))
        st.markdown("**Recent audit log**")
        st.dataframe(pd.DataFrame(auth.list_audit(50), columns=["timestamp", "user", "action", "detail"]))

st.divider()
st.caption(f"RepurposeAlpha · {disease} · {len(tickers)} candidates · Registry-driven")
