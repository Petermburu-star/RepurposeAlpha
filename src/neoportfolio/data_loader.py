"""
NeoPortfolio — NDD data loader.

Loads the Neoantigen Discovery Dataset (NDD v0.2) from Hugging Face.
257 curated positive neoantigen peptides, 25 cancer types, 46 HLA class I alleles.
License: CC-BY-4.0.
"""
from pathlib import Path
from typing import Optional
import pandas as pd

NDD_HF_URL = "https://huggingface.co/datasets/NeoDiscovery/NDD/resolve/main/data/v0.2/ndd_v0.2.tsv"

def load_ndd(cache_dir: Optional[Path] = None, force_download: bool = False) -> pd.DataFrame:
    """
    Load the NDD v0.2 dataset.
    Downloads once, caches locally.
    """
    import requests

    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "neoportfolio"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "ndd_v0.2.tsv"

    if cache_file.exists() and not force_download:
        print(f"✓ Loaded cached NDD: {cache_file.name}")
    else:
        print(f"→ Downloading NDD from Hugging Face...")
        r = requests.get(NDD_HF_URL, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"NDD download failed: HTTP {r.status_code}")
        cache_file.write_bytes(r.content)
        print(f"✓ Downloaded {len(r.content):,} bytes")

    df = pd.read_csv(cache_file, sep="\t")

    # Normalize column names to lowercase, strip whitespace
    df.columns = [c.strip().lower() for c in df.columns]

    # Required columns per NDD field schema
    required = ["pubmed_id", "patient_id", "gene", "mt_peptide", "wt_peptide",
                "length", "hla", "response_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"NDD missing required columns: {missing}. Available: {list(df.columns)}")

    # Clean
    df = df.dropna(subset=["mt_peptide", "hla"])
    df["mt_peptide"] = df["mt_peptide"].str.strip().str.upper()
    df["hla"] = df["hla"].str.strip().str.upper()
    df["length"] = df["mt_peptide"].str.len()

    # Standardize HLA format
    df["hla"] = df["hla"].str.replace("HLA-", "", regex=False)

    return df.reset_index(drop=True)

def filter_by_hla(df: pd.DataFrame, hla_alleles: list) -> pd.DataFrame:
    """Subset NDD to peptides restricted to any of the given HLA alleles."""
    alleles_upper = [a.upper().replace("HLA-", "") for a in hla_alleles]
    mask = df["hla"].apply(lambda x: any(a in str(x).upper() for a in alleles_upper))
    return df[mask].copy().reset_index(drop=True)

def sample_patient_cohort(df: pd.DataFrame, n_patients: int = 1,
                           random_state: int = 42) -> pd.DataFrame:
    """Extract a cohort of patients with their neoantigen peptides."""
    unique_patients = df["patient_id"].dropna().unique()
    if len(unique_patients) < n_patients:
        n_patients = len(unique_patients)
    selected = pd.Series(unique_patients).sample(n=n_patients, random_state=random_state).tolist()
    return df[df["patient_id"].isin(selected)].copy().reset_index(drop=True)
