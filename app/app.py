"""
RepurposeAlpha — Streamlit demo with RBAC.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from pypfopt import EfficientFrontier

import auth

st.set_page_config(page_title="RepurposeAlpha", page_icon="🧬", layout="wide")

user = auth.require_login(min_role="viewer")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_DIR / "pws_correlation_matrix.csv", index_col=0)

corr = load_data()
tickers = list(corr.columns)

col_title, col_user = st.columns([5, 1])
with col_title:
    st.title("🧬 RepurposeAlpha")
with col_user:
    st.caption(f"👤 {user['username']} ({user['role']})")
    if st.button("Log out"):
        auth.log_event(user["username"], "logout")
        st.session_state["user"] = None
        st.rerun()

st.markdown("**Portfolio optimization for drug repurposing.**")
st.divider()

st.sidebar.header("Assumptions")
default_returns = {"Carbetocin": 0.50, "Oxytocin": 0.40,
                   "Setmelanotide": 0.55, "Tirzepatide": 0.60}
default_vols = {"Carbetocin": 0.30, "Oxytocin": 0.25,
                "Setmelanotide": 0.35, "Tirzepatide": 0.40}

mu, sigma = {}, {}
for t in tickers:
    st.sidebar.markdown(f"**{t}**")
    mu[t] = st.sidebar.slider("Return", 0.0, 1.0, default_returns.get(t, 0.5), 0.01, key=f"mu_{t}")
    sigma[t] = st.sidebar.slider("Volatility", 0.05, 0.60, default_vols.get(t, 0.30), 0.01, key=f"sig_{t}")

mu_s = pd.Series(mu)
sigma_s = pd.Series(sigma)
cov = pd.DataFrame(np.outer(sigma_s, sigma_s) * corr.values,
                   index=tickers, columns=tickers)

def solve(mode):
    ef = EfficientFrontier(mu_s, cov, weight_bounds=(0, 1))
    try:
        if mode == "sharpe":
            ef.max_sharpe(risk_free_rate=0.02)
        else:
            ef.min_volatility()
        w = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.02)
        return pd.Series(w), ret, vol, sharpe
    except Exception as e:
        st.warning(f"Solver failed: {e}")
        return pd.Series(0.0, index=tickers), 0.0, 0.0, 0.0

w_sharpe, r_s, v_s, s_s = solve("sharpe")
w_minvol, r_m, v_m, _   = solve("minvol")

tab_names = ["🔗 Correlation", "📈 Efficient Frontier", "💼 Optimal Portfolio", "🎚️ Sensitivity"]
if auth.has_role(user, "admin"):
    tab_names.append("🛡️ Admin")

tabs = st.tabs(tab_names)
tab1, tab2, tab3, tab4 = tabs[0], tabs[1], tabs[2], tabs[3]

with tab1:
    st.subheader("Candidate correlation matrix")
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn_r",
                    zmin=0, zmax=1, aspect="auto")
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
    rho = st.slider("Carbetocin - Oxytocin correlation", 0.0, 1.0, 0.748, 0.01)
    corr2 = corr.copy()
    corr2.loc["Carbetocin", "Oxytocin"] = rho
    corr2.loc["Oxytocin", "Carbetocin"] = rho
    cov2 = pd.DataFrame(np.outer(sigma_s, sigma_s) * corr2.values,
                        index=tickers, columns=tickers)
    ef = EfficientFrontier(mu_s, cov2, weight_bounds=(0, 1))
    try:
        ef.max_sharpe(risk_free_rate=0.02)
        st.bar_chart(pd.Series(ef.clean_weights()))
    except Exception as e:
        st.error(f"Solver error: {e}")

if auth.has_role(user, "admin") and len(tabs) == 5:
    with tabs[4]:
        st.subheader("🛡️ Admin panel")
        st.markdown("**Users**")
        users = auth.list_users()
        st.dataframe(pd.DataFrame(users, columns=["username", "role", "created_at"]))

        st.markdown("**Recent audit log (50 events)**")
        audit = auth.list_audit(50)
        st.dataframe(pd.DataFrame(audit, columns=["timestamp", "user", "action", "detail"]))

        st.markdown("**Create new user**")
        with st.form("create_user_form"):
            new_user = st.text_input("Username")
            new_pw = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["viewer", "analyst", "admin"])
            submit = st.form_submit_button("Create")
        if submit and new_user and new_pw:
            ok = auth.create_user(new_user, new_pw, new_role)
            if ok:
                st.success(f"Created {new_user} ({new_role})")
            else:
                st.error("User already exists.")

st.divider()
st.caption("RepurposeAlpha · Phase 3.2.2 · RBAC-enabled")
