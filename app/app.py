"""
RepurposeAlpha — Streamlit demo
Portfolio optimization for drug repurposing.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from pypfopt import EfficientFrontier

# ---------- Config ----------
st.set_page_config(page_title="RepurposeAlpha", page_icon="🧬", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

# ---------- Data ----------
@st.cache_data
def load_data():
    corr = pd.read_csv(DATA_DIR / "pws_correlation_matrix.csv", index_col=0)
    return corr

corr = load_data()
tickers = list(corr.columns)

# ---------- Header ----------
st.title("🧬 RepurposeAlpha")
st.markdown(
    "**Portfolio optimization for drug repurposing.** "
    "Treating candidate drugs as correlated, risky assets — and allocating "
    "validation resources the way a hedge fund allocates capital."
)
st.divider()

# ---------- Sidebar: assumptions ----------
st.sidebar.header("Assumptions")
st.sidebar.caption("Move the sliders — the portfolio updates live.")

default_returns = {
    "Carbetocin": 0.50, "Oxytocin": 0.40,
    "Setmelanotide": 0.55, "Tirzepatide": 0.60,
}
default_vols = {
    "Carbetocin": 0.30, "Oxytocin": 0.25,
    "Setmelanotide": 0.35, "Tirzepatide": 0.40,
}

mu, sigma = {}, {}
for t in tickers:
    st.sidebar.markdown(f"**{t}**")
    mu[t] = st.sidebar.slider(f"Return (rNPV proxy)", 0.0, 1.0,
                              default_returns.get(t, 0.5), 0.01, key=f"mu_{t}")
    sigma[t] = st.sidebar.slider(f"Volatility", 0.05, 0.60,
                                 default_vols.get(t, 0.30), 0.01, key=f"sig_{t}")

mu_s = pd.Series(mu)
sigma_s = pd.Series(sigma)

cov = pd.DataFrame(
    np.outer(sigma_s, sigma_s) * corr.values,
    index=tickers, columns=tickers
)

# ---------- Portfolio calc ----------
def solve(mode):
    ef = EfficientFrontier(mu_s, cov, weight_bounds=(0, 1))
    try:
        if mode == "sharpe":
            ef.max_sharpe(risk_free_rate=0.02)
        else:
            ef.min_volatility()
        weights = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.02)
        return pd.Series(weights), ret, vol, sharpe
    except Exception as e:
        st.warning(f"Solver failed: {e}")
        return pd.Series(0.0, index=tickers), 0.0, 0.0, 0.0

w_sharpe, r_s, v_s, s_s = solve("sharpe")
w_minvol, r_m, v_m, _   = solve("minvol")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔗 Correlation", "📈 Efficient Frontier", "💼 Optimal Portfolio", "🎚️ Sensitivity"]
)

# ----- Tab 1: correlation -----
with tab1:
    st.subheader("Candidate correlation matrix")
    st.markdown(
        "**1.0 = identical bet · 0.0 = independent.** "
        "The Carbetocin–Oxytocin pair at ~0.75 means the same oxytocin/vasopressin "
        "hypothesis is being tested twice."
    )
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn_r",
                    zmin=0, zmax=1, aspect="auto")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

# ----- Tab 2: efficient frontier -----
with tab2:
    st.subheader("Efficient frontier")
    st.markdown("Each point = a portfolio. The curve = best risk-adjusted options.")
    # Monte Carlo random portfolios
    n = 3000
    rand_w = np.random.dirichlet(np.ones(len(tickers)), n)
    rets = rand_w @ mu_s.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, cov.values, rand_w))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vols, y=rets, mode="markers",
        marker=dict(size=4, color=rets/vols, colorscale="Viridis",
                    colorbar=dict(title="Sharpe")),
        name="Random portfolios"
    ))
    fig.add_trace(go.Scatter(
        x=[v_s, v_m], y=[r_s, r_m], mode="markers+text",
        marker=dict(size=18, color=["gold", "silver"],
                    line=dict(color="black", width=1)),
        text=["Max Sharpe", "Min Vol"], textposition="top center",
        name="Optimal portfolios"
    ))
    fig.update_layout(xaxis_title="Risk (volatility)", yaxis_title="Expected return",
                      height=520)
    st.plotly_chart(fig, use_container_width=True)

# ----- Tab 3: optimal portfolio -----
with tab3:
    st.subheader("Optimal portfolio weights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Max Sharpe**")
        st.dataframe(w_sharpe.rename("weight").to_frame().style.format("{:.3f}"))
        st.metric("Expected return", f"{r_s:.3f}")
        st.metric("Volatility", f"{v_s:.3f}")
        st.metric("Sharpe ratio", f"{s_s:.3f}")
    with col2:
        st.markdown("**Min Volatility**")
        st.dataframe(w_minvol.rename("weight").to_frame().style.format("{:.3f}"))
        st.metric("Expected return", f"{r_m:.3f}")
        st.metric("Volatility", f"{v_m:.3f}")

    fig = px.bar(
        pd.DataFrame({"Max Sharpe": w_sharpe, "Min Vol": w_minvol}),
        barmode="group", title="Weights by strategy"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----- Tab 4: sensitivity -----
with tab4:
    st.subheader("How sensitive is the portfolio to the correlation assumption?")
    st.markdown(
        "Sweeping the Carbetocin–Oxytocin correlation from 0 to 1. "
        "Notice the non-monotonic behavior around rho = 0.7."
    )
    rho = st.slider("Carbetocin ↔ Oxytocin correlation", 0.0, 1.0, 0.748, 0.01)
    # Rebuild correlation with the chosen rho
    corr2 = corr.copy()
    corr2.loc["Carbetocin", "Oxytocin"] = rho
    corr2.loc["Oxytocin", "Carbetocin"] = rho
    cov2 = pd.DataFrame(np.outer(sigma_s, sigma_s) * corr2.values,
                        index=tickers, columns=tickers)
    ef = EfficientFrontier(mu_s, cov2, weight_bounds=(0, 1))
    try:
        ef.max_sharpe(risk_free_rate=0.02)
        w = pd.Series(ef.clean_weights())
        st.bar_chart(w)
        st.caption(f"At rho = {rho:.2f}, the optimizer allocates as shown above.")
    except Exception as e:
        st.error(f"Solver error: {e}")

# ---------- Footer ----------
st.divider()
st.caption(
    "RepurposeAlpha · Phase 3.1 demo · "
    "Data: RepoDB + ChEMBL (public) · Framework: Markowitz via PyPortfolioOpt · "
    "github.com/Petermburu-star/RepurposeAlpha"
)
