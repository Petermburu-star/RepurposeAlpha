"""
RepurposeAlpha — assumption registry.

Every numeric input to the tool is stored here with a source, confidence
label, and timestamp. Nothing is hardcoded in the app.

Storage: YAML files in config/assumptions/
Confidence tiers: high / medium / low
"""
from pathlib import Path
from datetime import datetime, timezone
import yaml
import requests


APP_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = APP_DIR / "config" / "assumptions"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = CONFIG_DIR / "registry.yaml"


def load_registry() -> dict:
    """Load the entire registry from YAML."""
    if not REGISTRY_FILE.exists():
        return {}
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_registry(registry: dict) -> None:
    """Write the registry back to YAML."""
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f, sort_keys=False, allow_unicode=True)


def get_value(key: str, default=None):
    """Retrieve a single value from the registry by dotted key path."""
    reg = load_registry()
    parts = key.split(".")
    node = reg
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    if isinstance(node, dict) and "value" in node:
        return node["value"]
    return node


def set_value(key: str, value, source: str, confidence: str = "medium",
               method: str = "manual", notes: str = "") -> None:
    """Set a value with metadata."""
    if confidence not in ("high", "medium", "low"):
        raise ValueError(f"Confidence must be high/medium/low, got {confidence}")

    reg = load_registry()
    parts = key.split(".")
    node = reg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = {
        "value": value,
        "source": source,
        "confidence": confidence,
        "method": method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }
    save_registry(reg)


def fetch_risk_free_rate() -> dict:
    """Fetch live 10-year US Treasury yield from FRED (free, no key)."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    try:
        r = requests.get(url, timeout=15)
        lines = r.text.strip().split("\n")
        latest = lines[-1].split(",")
        rate = float(latest[1]) / 100.0
        return {
            "value": rate,
            "source": "FRED DGS10 (10-year US Treasury)",
            "confidence": "high",
            "method": "live_fetch",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": f"Latest observation date: {latest[0]}",
        }
    except Exception as e:
        return {
            "value": 0.02,
            "source": "fallback default",
            "confidence": "low",
            "method": "fallback",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": f"FRED fetch failed: {e}",
        }


def refresh_live_values() -> dict:
    """Refresh all values that should come from live sources."""
    updates = {"risk_free_rate": fetch_risk_free_rate()}
    reg = load_registry()
    reg.update(updates)
    save_registry(reg)
    return updates
