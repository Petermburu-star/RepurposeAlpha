"""
RepurposeAlpha — multi-signal adaptive correlation engine v4.

Seven signals, adaptively weighted based on data availability.
Works for receptor-targeted AND organism-targeted drugs.
"""
import requests
import pandas as pd
import numpy as np
import time
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

BINDING_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50"}
UNIT_TO_NM = {
    "nM": 1.0, "uM": 1000.0, "µM": 1000.0, "μM": 1000.0,
    "mM": 1_000_000.0, "pM": 0.001, "M": 1_000_000_000.0,
}
AFFINITY_THRESHOLD_NM = 10_000.0

SPECIES_BLOCKLIST = {
    "Rattus norvegicus", "Homo sapiens", "Mus musculus",
    "Escherichia coli", "Klebsiella pneumoniae", "Bacillus subtilis",
    "Plasmodium falciparum", "Plasmodium cynomolgi", "Plasmodium vivax",
    "Toxoplasma gondii", "Trypanosoma brucei", "Trypanosoma cruzi",
    "Leishmania", "Trichomonas vaginalis", "Pneumocystis carinii",
    "SARS-CoV-2", "Gallus gallus", "Bos taurus", "Sus scrofa",
    "Saccharomyces cerevisiae", "Mycobacterium tuberculosis",
    "Staphylococcus aureus", "Candida albicans", "Unchecked",
    "No relevant target", "Hepatotoxicity", "ADMET", "THP-1",
}

STOPWORDS = {
    "of", "in", "the", "to", "a", "and", "or", "for", "on", "at",
    "by", "with", "activity", "assay", "inhibition", "inhibitor",
    "against", "human", "using", "based", "results", "test", "tested",
}


# ---------- LOW-LEVEL FETCHERS ----------

def fetch_activities(chembl_id: str, timeout: int = 90, max_retries: int = 3) -> pd.DataFrame:
    url = f"{CHEMBL_BASE}/activity.json"
    params = {"molecule_chembl_id": chembl_id, "limit": 500}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                rows = []
                for act in r.json().get("activities", []):
                    rows.append({
                        "chembl_id": chembl_id,
                        "target_chembl_id": act.get("target_chembl_id"),
                        "target_name": act.get("target_pref_name"),
                        "target_organism": act.get("target_organism"),
                        "standard_type": act.get("standard_type"),
                        "standard_value": act.get("standard_value"),
                        "standard_units": act.get("standard_units"),
                        "assay_description": (act.get("assay_description") or "")[:300],
                    })
                return pd.DataFrame(rows)
            elif r.status_code == 500:
                time.sleep(2 ** attempt); continue
            else:
                return pd.DataFrame()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt); continue
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def fetch_mechanism(chembl_id: str, timeout: int = 30) -> dict:
    """Fetch MoA records for a drug."""
    try:
        r = requests.get(f"{CHEMBL_BASE}/mechanism.json",
                         params={"molecule_chembl_id": chembl_id, "limit": 20}, timeout=timeout)
        if r.status_code == 200:
            mechs = r.json().get("mechanisms", [])
            moa_texts = [m.get("mechanism_of_action") or "" for m in mechs]
            return {
                "moa": " ".join(moa_texts).lower(),
                "action_types": [m.get("action_type") or "" for m in mechs],
            }
    except Exception:
        pass
    return {"moa": "", "action_types": []}


def fetch_molecule(chembl_id: str, timeout: int = 30) -> dict:
    """Fetch SMILES + ATC codes."""
    try:
        r = requests.get(f"{CHEMBL_BASE}/molecule/{chembl_id}.json", timeout=timeout)
        if r.status_code == 200:
            mol = r.json()
            return {
                "smiles": (mol.get("molecule_structures") or {}).get("canonical_smiles") or "",
                "atc_codes": mol.get("atc_classifications") or [],
            }
    except Exception:
        pass
    return {"smiles": "", "atc_codes": []}


# ---------- SIMILARITY PRIMITIVES ----------

def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def tokenize(text: str) -> set:
    if not text:
        return set()
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def moa_similarity(moa1: str, moa2: str) -> float:
    return jaccard(tokenize(moa1), tokenize(moa2))


def assay_desc_similarity(descs1: list, descs2: list) -> float:
    """Compare two sets of assay descriptions via token overlap."""
    if not descs1 or not descs2:
        return 0.0
    tokens1 = set()
    for d in descs1[:20]:  # sample first 20 to keep it fast
        tokens1 |= tokenize(d)
    tokens2 = set()
    for d in descs2[:20]:
        tokens2 |= tokenize(d)
    return jaccard(tokens1, tokens2)


def tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    if not smiles1 or not smiles2:
        return 0.0
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator, DataStructs
        m1 = Chem.MolFromSmiles(smiles1)
        m2 = Chem.MolFromSmiles(smiles2)
        if m1 is None or m2 is None:
            return 0.0
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return float(DataStructs.TanimotoSimilarity(gen.GetFingerprint(m1), gen.GetFingerprint(m2)))
    except Exception:
        return 0.0


def murcko_scaffold(smiles: str) -> str:
    """Return canonical SMILES of the Murcko scaffold, or '' on failure."""
    if not smiles:
        return ""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


def atc_similarity(atc1: list, atc2: list) -> float:
    """Hierarchical ATC prefix match. Best score across all code pairs."""
    if not atc1 or not atc2:
        return 0.0
    best = 0.0
    for a in atc1:
        for b in atc2:
            a, b = a.upper(), b.upper()
            if len(a) < 3 or len(b) < 3:
                continue
            prefix_len = 0
            for i in range(min(len(a), len(b))):
                if a[i] == b[i]:
                    prefix_len += 1
                else:
                    break
            # Prefix -> score: level 1 (letter) = 0.2, up to 7 chars = 1.0
            score = min(prefix_len / 5.0, 1.0) * 0.9 + (0.1 if prefix_len >= 3 else 0)
            best = max(best, score)
    return best


# ---------- ACTIVITY TABLE ----------

def build_activity_table(chembl_ids: list, verbose: bool = True,
                         max_workers: int = 4):
    if verbose:
        print(f"  Fetching {len(chembl_ids)} drugs (activities + MoA + molecule)...")

    activities_by_drug = {}
    moa_map = {}
    molecule_map = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Submit all three fetches per drug
        futs = {}
        for cid in chembl_ids:
            futs[ex.submit(fetch_activities, cid)] = ("act", cid)
            futs[ex.submit(fetch_mechanism, cid)] = ("moa", cid)
            futs[ex.submit(fetch_molecule, cid)] = ("mol", cid)

        for fut in as_completed(futs):
            kind, cid = futs[fut]
            try:
                result = fut.result()
            except Exception:
                result = None
            if kind == "act" and result is not None and not result.empty:
                activities_by_drug[cid] = result
            elif kind == "moa" and result:
                moa_map[cid] = result
            elif kind == "mol" and result:
                molecule_map[cid] = result

    if not activities_by_drug:
        return pd.DataFrame(), moa_map, molecule_map

    combined = pd.concat(activities_by_drug.values(), ignore_index=True)
    combined = combined[~combined["target_name"].isin(SPECIES_BLOCKLIST)]
    combined = combined[combined["target_chembl_id"].notna()]
    combined["standard_value"] = pd.to_numeric(combined["standard_value"], errors="coerce")
    combined = combined[combined["standard_type"].isin(BINDING_TYPES)]
    combined = combined.dropna(subset=["standard_value"])
    combined["value_nM"] = (
        combined["standard_value"]
        * combined["standard_units"].map(UNIT_TO_NM).fillna(1.0)
    )
    combined = combined[combined["value_nM"] <= AFFINITY_THRESHOLD_NM]
    return combined, moa_map, molecule_map


# ---------- MAIN ----------

def build_correlation_matrix(chembl_ids: list, verbose: bool = True) -> pd.DataFrame:
    n = len(chembl_ids)
    if n < 2:
        return pd.DataFrame()

    activities, moa_map, molecule_map = build_activity_table(chembl_ids, verbose)

    # --- Build per-drug features ---
    smiles_map = {cid: molecule_map.get(cid, {}).get("smiles", "") for cid in chembl_ids}
    atc_map = {cid: molecule_map.get(cid, {}).get("atc_codes", []) for cid in chembl_ids}
    moa_text = {cid: moa_map.get(cid, {}).get("moa", "") for cid in chembl_ids}
    scaffold_map = {cid: murcko_scaffold(smiles_map.get(cid, "")) for cid in chembl_ids}

    # Assay descriptions per drug
    assay_descs = {cid: [] for cid in chembl_ids}
    if not activities.empty:
        for cid, grp in activities.groupby("chembl_id"):
            assay_descs[cid] = grp["assay_description"].dropna().tolist()[:20]

    # Binary target matrix
    if not activities.empty:
        target_ids = sorted(activities["target_chembl_id"].unique())
        binary = pd.DataFrame(0, index=chembl_ids, columns=target_ids)
        for _, r in activities.iterrows():
            binary.loc[r["chembl_id"], r["target_chembl_id"]] = 1
        median_aff = activities.groupby(
            ["chembl_id", "target_chembl_id"]
        )["value_nM"].median().to_dict()
    else:
        binary = pd.DataFrame()
        target_ids = []
        median_aff = {}

    # --- Data availability audit (for adaptive weighting) ---
    n_with_targets = sum(1 for cid in chembl_ids if cid in activities["chembl_id"].values if not activities.empty) if not activities.empty else 0
    n_with_atc    = sum(1 for cid in chembl_ids if atc_map.get(cid))
    n_with_moa    = sum(1 for cid in chembl_ids if moa_text.get(cid))
    n_with_smiles = sum(1 for cid in chembl_ids if smiles_map.get(cid))

    if verbose:
        print(f"  Data availability:")
        print(f"    targets:  {n_with_targets}/{n} drugs")
        print(f"    ATC:      {n_with_atc}/{n} drugs")
        print(f"    MoA:      {n_with_moa}/{n} drugs")
        print(f"    SMILES:   {n_with_smiles}/{n} drugs")

    # --- Adaptive weights ---
    weights = {
        "target":   0.35 if n_with_targets / n >= 0.5 else 0.05,
        "affinity": 0.15 if n_with_targets / n >= 0.5 else 0.05,
        "structure": 0.25,
        "scaffold":  0.10,
        "moa":       0.10 if n_with_moa / n >= 0.3 else 0.05,
        "atc":       0.15 if n_with_atc / n >= 0.3 else 0.05,
        "assay":     0.10,
    }
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}  # normalize

    if verbose:
        print(f"  Adaptive weights: " +
              ", ".join(f"{k}={v:.2f}" for k, v in weights.items()))

    # --- Pairwise ---
    def affinity_sim(d1, d2):
        if not target_ids:
            return 0.0
        a, b = binary.loc[d1].values, binary.loc[d2].values
        shared = np.array(target_ids)[(a == 1) & (b == 1)]
        if shared.size == 0:
            return 0.0
        sims = []
        for t in shared:
            a1, a2 = median_aff.get((d1, t)), median_aff.get((d2, t))
            if pd.notna(a1) and pd.notna(a2) and a1 > 0 and a2 > 0:
                sims.append(np.exp(-abs(np.log10(a1) - np.log10(a2))))
        return float(np.mean(sims)) if sims else 0.0

    mat = pd.DataFrame(np.zeros((n, n)), index=chembl_ids, columns=chembl_ids)
    for i, d1 in enumerate(chembl_ids):
        for j, d2 in enumerate(chembl_ids):
            if d1 == d2:
                mat.loc[d1, d2] = 1.0
                continue

            s_target    = jaccard(set(binary.loc[d1][binary.loc[d1] == 1].index),
                                  set(binary.loc[d2][binary.loc[d2] == 1].index)) if target_ids else 0.0
            s_affinity  = affinity_sim(d1, d2)
            s_structure = tanimoto_similarity(smiles_map.get(d1, ""), smiles_map.get(d2, ""))
            s_scaffold  = 1.0 if (scaffold_map.get(d1) and scaffold_map.get(d1) == scaffold_map.get(d2)) else 0.0
            s_moa       = moa_similarity(moa_text.get(d1, ""), moa_text.get(d2, ""))
            s_atc       = atc_similarity(atc_map.get(d1, []), atc_map.get(d2, []))
            s_assay     = assay_desc_similarity(assay_descs.get(d1, []), assay_descs.get(d2, []))

            score = (
                weights["target"]    * s_target +
                weights["affinity"]  * s_affinity +
                weights["structure"] * s_structure +
                weights["scaffold"]  * s_scaffold +
                weights["moa"]       * s_moa +
                weights["atc"]       * s_atc +
                weights["assay"]     * s_assay
            )
            mat.loc[d1, d2] = score

    if verbose:
        print(f"  ✓ Matrix {n}×{n} | targets: {len(target_ids)} | records: {len(activities)}")
    return mat
