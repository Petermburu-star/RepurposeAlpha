"""
RepurposeAlpha — authentication + role-based access control.
"""
import sqlite3
import bcrypt
import streamlit as st
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "users.db"

ROLE_LEVELS = {"viewer": 1, "analyst": 2, "admin": 3}


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username   TEXT PRIMARY KEY,
                pw_hash    BLOB NOT NULL,
                role       TEXT NOT NULL DEFAULT 'analyst',
                created_at TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT NOT NULL,
                username   TEXT,
                action     TEXT NOT NULL,
                detail     TEXT
            )
        """)


def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))


def verify_password(password: str, pw_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), pw_hash)
    except Exception:
        return False


def create_user(username: str, password: str, role: str = "analyst") -> bool:
    if role not in ROLE_LEVELS:
        raise ValueError(f"Unknown role: {role}. Must be one of {list(ROLE_LEVELS)}")
    try:
        with _conn() as con:
            con.execute(
                "INSERT INTO users (username, pw_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (username, hash_password(password), role, datetime.utcnow().isoformat()),
            )
        log_event(username, "create_user", f"role={role}")
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate(username: str, password: str):
    with _conn() as con:
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
    with _conn() as con:
        con.execute(
            "INSERT INTO audit_log (ts, username, action, detail) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), username, action, detail),
        )


def list_users():
    with _conn() as con:
        return con.execute(
            "SELECT username, role, created_at FROM users ORDER BY created_at"
        ).fetchall()


def list_audit(limit=50):
    with _conn() as con:
        return con.execute(
            "SELECT ts, username, action, detail FROM audit_log "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


def has_role(user, required_role: str) -> bool:
    if not user:
        return False
    return ROLE_LEVELS.get(user.get("role"), 0) >= ROLE_LEVELS.get(required_role, 999)


def require_login(min_role: str = "viewer"):
    """Streamlit guard with minimum role requirement."""
    if "user" in st.session_state and st.session_state["user"]:
        if has_role(st.session_state["user"], min_role):
            return st.session_state["user"]
        st.error(f"Your role ({st.session_state['user']['role']}) cannot access this page. "
                 f"Required: {min_role}.")
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
