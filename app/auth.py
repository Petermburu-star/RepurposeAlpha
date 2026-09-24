"""
RepurposeAlpha — authentication module.
Passwords hashed with bcrypt, sessions managed via Streamlit state.
"""
import sqlite3
import bcrypt
import streamlit as st
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "users.db"


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create users and audit_log tables if they don't exist."""
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
    """Hash a password with bcrypt (cost factor 12)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))


def verify_password(password: str, pw_hash: bytes) -> bool:
    """Verify a plaintext password against a stored hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), pw_hash)
    except Exception:
        return False


def create_user(username: str, password: str, role: str = "analyst") -> bool:
    """Create a user. Returns True on success, False if user exists."""
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
    """Check credentials. Returns user dict or None."""
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
    """Append to audit log."""
    with _conn() as con:
        con.execute(
            "INSERT INTO audit_log (ts, username, action, detail) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), username, action, detail),
        )


def require_login():
    """
    Streamlit guard. Returns the logged-in user dict, or renders login
    form and stops execution until authenticated.
    """
    if "user" in st.session_state and st.session_state["user"]:
        return st.session_state["user"]

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
            st.success(f"Welcome, {user['username']}")
            st.rerun()
        else:
            st.error("Invalid credentials.")
    st.stop()
