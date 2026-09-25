"""
RepurposeAlpha — authentication + RBAC (encrypted storage).
"""
import bcrypt
import sqlite3
from datetime import datetime

import secure_db
from secure_db import connect

ROLE_LEVELS = {"viewer": 1, "analyst": 2, "admin": 3}


def init_db():
    secure_db.initialize()


def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))


def verify_password(password: str, pw_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), pw_hash)
    except Exception:
        return False


def create_user(username: str, password: str, role: str = "analyst") -> bool:
    if role not in ROLE_LEVELS:
        raise ValueError(f"Unknown role: {role}")
    try:
        with connect() as con:
            con.execute(
                "INSERT INTO users (username, pw_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (username, hash_password(password), role, datetime.utcnow().isoformat()),
            )
        log_event(username, "create_user", f"role={role}")
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate(username: str, password: str):
    with connect() as con:
        row = con.execute(
            "SELECT username, pw_hash, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row and verify_password(password, row[1]):
        log_event(username, "login_success")
        return {"username": row[0], "role": row[2]}
    log_event(username, "login_failure")
    return None


def log_event(username, action, detail=None):
    with connect() as con:
        con.execute(
            "INSERT INTO audit_log (ts, username, action, detail) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), username, action, detail),
        )


def list_users():
    with connect() as con:
        return con.execute(
            "SELECT username, role, created_at FROM users ORDER BY created_at"
        ).fetchall()


def list_audit(limit=50):
    with connect() as con:
        return con.execute(
            "SELECT ts, username, action, detail FROM audit_log "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


def has_role(user, required_role: str) -> bool:
    if not user:
        return False
    return ROLE_LEVELS.get(user.get("role"), 0) >= ROLE_LEVELS.get(required_role, 999)


def require_login(min_role: str = "viewer"):
    import streamlit as st
    if "user" in st.session_state and st.session_state["user"]:
        if has_role(st.session_state["user"], min_role):
            return st.session_state["user"]
        st.error(f"Your role cannot access this page.")
        st.stop()

    st.title("🔒 RepurposeAlpha")
    st.caption("Authentication required.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Log in")
    if submit:
        user = authenticate(username, password)
        if user:
            st.session_state["user"] = user
            st.rerun()
        else:
            st.error("Invalid credentials.")
    st.stop()
