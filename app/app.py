"""
RepurposeAlpha — Streamlit app (registry-driven).
"""
import sys
from pathlib import Path

# Make src/ importable BEFORE importing our modules
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

st.set_page_config(page_title="RepurposeAlpha", page_icon="🧬", layout="wide")

# ---------- Auth gate ----------
user = auth.require_login(min_role="viewer")

# ---------- Data ----------
DATA_DIR = ROOT / "data" / "processed"

@st.cache_data
def load_corr():
    return pd.read_csv(DATA_DIR / "pws_correlation_matrix.csv", index_col=0)

corr = load_corr()
tickers = list(corr.columns)

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

st.markdown("**Portfolio optimization for drug repurposing.** All inputs sourced from the assumption registry.")
st.divider()

# ---------- Sidebar ----------
st.sidebar.header("Assumptions")
st.sidebar.caption("Defaults from registry · every value shows its source")

rfr_entry = assumptions.load_registry().get("risk_free_rate", {})
rfr_value  = rfr_entry.get("value", 0.02)
rfr_source = rfr_entry.get("source", "unknown")

st.sidebar.markdown(f"**Risk-free rate:** `{rfr_value:.4f}`")
st.sidebar.caption(f"Source: {rfr_source}")

pos_phase2 = assumptions.get_value("pos.phase_2", 0.31)
pos_phase3 = assumptions.get_value("pos.phase_3", 0.58)

st.sidebar.markdown("---")
st.sidebar.markdown("**Per-candidate estimates**")
st.sidebar.caption(f"PoS baseline from BIO/QLS (Phase 2 = {pos_phase2:.2f}, Phase 3 = {pos_phase3:.2f})")

default_returns = {t: round(pos_phase3, 2) for t in tickers}
default_vols    = {t: 0.30 for t in tickers}

mu, sigma = {}, {}
for t in tickers:
    st.sidebar.markdown(f"**{t}**")
    mu[t] = st.sidebar.slider("Return (rNPV proxy)", 0.0, 1.0, default_returns[t], 0.01, key=f"mu_{t}")
    sigma[t] = st.sidebar.slider("Volatility", 0.05, 0.60, default_vols[t], 0.01, key=f"sig_{t}")

mu_s = pd.Series(mu)
sigma_s = pd.Series(sigma)
cov = pd.DataFrame(np.outer(sigma_s, sigma_s) * corr.values, index=tickers, columns=tickers)

# ---------- Solver ----------
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

# ---------- Tabs ----------
tab_names = ["🔗 Correlation", "📈 Efficient Frontier", "💼 Optimal Portfolio", "🎚️ Sensitivity", "📋 Assumptions"]
if auth.has_role(user, "admin"):
    tab_names.append("🛡️ Admin")

tabs = st.tabs(tab_names)
tab1, tab2, tab3, tab4, tab5 = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4]

with tab1:
    st.subheader("Candidate correlation matrix")
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn_r", zmin=0, zmax=1, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Efficient frontier")
    n = 3000
    rand_w = np.random.dirichlet(np.ones(len(tickers)), n)
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

with tab3:
    st.subheader("Optimal portfolio weights")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Max Sharpe**")
        st.dataframe(w_sharpe.rename("weight").to_frame().style.format("{:.3f}"))
        st.metric("Expected return", f"{r_s:.3f}")
        st.metric("Volatility", f"{v_s:.3f}")
        st.metric("Sharpe ratio", f"{s_s:.3f}")
    with c2:
        st.markdown("**Min Volatility**")
        st.dataframe(w_minvol.rename("weight").to_frame().style.format("{:.3f}"))
        st.metric("Expected return", f"{r_m:.3f}")
        st.metric("Volatility", f"{v_m:.3f}")

with tab4:
    st.subheader("Sensitivity to correlation assumption")
    rho = st.slider("Carbetocin ↔ Oxytocin correlation", 0.0, 1.0, 0.748, 0.01)
    corr2 = corr.copy()
    if "Carbetocin" in corr2.index and "Oxytocin" in corr2.index:
        corr2.loc["Carbetocin", "Oxytocin"] = rho
        corr2.loc["Oxytocin", "Carbetocin"] = rho
    cov2 = pd.DataFrame(np.outer(sigma_s, sigma_s) * corr2.values, index=tickers, columns=tickers)
    ef = EfficientFrontier(mu_s, cov2, weight_bounds=(0, 1))
    try:
        ef.max_sharpe(risk_free_rate=rfr_value)
        st.bar_chart(pd.Series(ef.clean_weights()))
    except Exception as e:
        st.error(f"Solver error: {e}")

with tab5:
    st.subheader("📋 Assumption Registry")
    st.caption("Every numeric input to the model, with source, confidence, and timestamp.")
    reg = assumptions.load_registry()

    rows = []
    def flatten(node, prefix=""):
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict) and "value" in v:
                rows.append({
                    "key": path,
                    "value": v["value"],
                    "confidence": v.get("confidence", "?"),
                    "source": v.get("source", ""),
                    "timestamp": (v.get("timestamp") or "")[:19],
                })
            elif isinstance(v, dict):
                flatten(v, path)
    flatten(reg)

    df_reg = pd.DataFrame(rows)
    st.dataframe(df_reg, use_container_width=True, hide_index=True)
    conf_counts = df_reg["confidence"].value_counts().to_dict()
    st.markdown(f"**Confidence breakdown:** {conf_counts}")

if auth.has_role(user, "admin") and len(tabs) == 6:
    with tabs[5]:
        st.subheader("🛡️ Admin panel")
        st.markdown("**Users**")
        st.dataframe(pd.DataFrame(auth.list_users(), columns=["username", "role", "created_at"]))
        st.markdown("**Recent audit log**")
        st.dataframe(pd.DataFrame(auth.list_audit(50), columns=["timestamp", "user", "action", "detail"]))

st.divider()
st.caption("RepurposeAlpha · Phase 5.5 · Registry-driven")
