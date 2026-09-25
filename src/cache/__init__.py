"""
RepurposeAlpha — analysis cache.

Caches disease analyses so re-querying the same disease is instant.
Key: disease name (slugified).
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _key(disease_name: str) -> str:
    slug = disease_name.lower().strip().replace(" ", "_")
    return hashlib.md5(slug.encode()).hexdigest()[:12] + "_" + slug[:40]


def cache_path(disease_name: str) -> Path:
    return CACHE_DIR / f"{_key(disease_name)}"


def has_cache(disease_name: str) -> bool:
    p = cache_path(disease_name)
    return (p / "candidates.csv").exists() and (p / "correlation.csv").exists()


def save_analysis(disease_name: str, candidates, correlation, metadata=None):
    """Save candidates + correlation matrix to cache."""
    p = cache_path(disease_name)
    p.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(p / "candidates.csv", index=False)
    correlation.to_csv(p / "correlation.csv")
    meta = {
        "disease": disease_name,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "n_candidates": int(len(candidates)),
        **(metadata or {}),
    }
    (p / "metadata.json").write_text(json.dumps(meta, indent=2))


def load_analysis(disease_name: str):
    """Load cached candidates + correlation. Returns (candidates, correlation, metadata)."""
    import pandas as pd
    p = cache_path(disease_name)
    if not (p / "candidates.csv").exists():
        return None, None, None
    candidates = pd.read_csv(p / "candidates.csv")
    correlation = pd.read_csv(p / "correlation.csv", index_col=0)
    meta = {}
    if (p / "metadata.json").exists():
        meta = json.loads((p / "metadata.json").read_text())
    return candidates, correlation, meta


def list_cached() -> list:
    """List all cached analyses."""
    out = []
    for d in CACHE_DIR.iterdir():
        if d.is_dir() and (d / "metadata.json").exists():
            try:
                meta = json.loads((d / "metadata.json").read_text())
                out.append(meta)
            except Exception:
                pass
    return sorted(out, key=lambda x: x.get("cached_at", ""), reverse=True)
