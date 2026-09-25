"""
RepurposeAlpha — encrypted SQLite wrapper.
"""
import os
import sqlite3
from pathlib import Path
from cryptography.fernet import Fernet
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
ENC_PATH = APP_DIR / "users.db.enc"
PLAIN_PATH = APP_DIR / "users.db"

load_dotenv(APP_DIR.parent / ".env")

def _get_fernet():
    key = os.getenv("REPURPOSE_FERNET_KEY")
    if not key:
        raise RuntimeError("REPURPOSE_FERNET_KEY missing from .env.")
    return Fernet(key.encode("utf-8"))

def _decrypt_if_needed():
    f = _get_fernet()
    if ENC_PATH.exists():
        decrypted = f.decrypt(ENC_PATH.read_bytes())
        PLAIN_PATH.write_bytes(decrypted)

def _encrypt_on_save():
    f = _get_fernet()
    if PLAIN_PATH.exists():
        ENC_PATH.write_bytes(f.encrypt(PLAIN_PATH.read_bytes()))
        PLAIN_PATH.unlink()

class SecureConnection:
    def __init__(self):
        _decrypt_if_needed()
        self._con = sqlite3.connect(PLAIN_PATH)
    def __enter__(self):
        return self._con
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._con.commit()
        self._con.close()
        _encrypt_on_save()

def connect():
    return SecureConnection()

def initialize():
    with connect() as con:
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
