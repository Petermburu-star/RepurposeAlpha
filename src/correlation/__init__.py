"""
RepurposeAlpha — multi-signal adaptive correlation engine v5.

Adds entropy-based weight selection. Weights are chosen by the data
itself based on each signal's information content for the given drug set.
"""
import requests
import pandas as pd
import numpy as np
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from .weighting import compute_weights

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


# ---------------- Low-level fetchers ----------------

def fetch_activities(chembl_id, timeout=90, max_retries=3):
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


def fetch_mechanism(chembl_id, timeout=30):
    try:
        r = requests.get(f"{CHEMBL_BASE}/mechanism.json",
                         params={"molecule_chembl_id": chembl_id, "limit": 20}, timeout=timeout)
        if r.status_code == 200:
            mechs = r.json().get("mechanisms", [])
            return {"moa": " ".join((m.get("mechanism_of_action") or "") for m in mechs).lower()}
    except Exception:
        pass
    return {"moa": ""}


def fetch_molecule(chembl_id, timeout=30):
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


# ---------------- Similarity primitives ----------------

def jaccard(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def tokenize(text):
    if not text: return set()
    return {t for t in re.findall(r"[a-z0-9]+", text.lower())
            if len(t) > 2 and t not in STOPWORDS}


def moa_similarity(m1, m2):
    return jaccard(tokenize(m1), tokenize(m2))


def assay_similarity(d1, d2):
    if not d1 or not d2: return 0.0
    t1, t2 = set(), set()
    for x in d1[:20]: t1 |= tokenize(x)
    for x in d2[:20]: t2 |= tokenize(x)
    return jaccard(t1, t2)


def tanimoto(s1, s2):
    if not s1 or not s2: return 0.0
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator, DataStructs
        m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
        if not m1 or not m2: return 0.0
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return float(DataStructs.TanimotoSimilarity(gen.GetFingerprint(m1), gen.GetFingerprint(m2)))
    except Exception:
        return 0.0


def murcko(smiles):
    if not smiles: return ""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return ""
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return ""


def atc_sim(a1, a2):
    if not a1 or not a2: return 0.0
    best = 0.0
    for a in a1:
        for b in a2:
            a, b = a.upper(), b.upper()
            pre = 0
            for i in range(min(len(a), len(b))):
                if a[i] == b[i]: pre += 1
                else: break
            best = max(best, min(pre / 5.0, 1.0) * 0.9 + (0.1 if pre >= 3 else 0))
    return best


# ---------------- Main engine ----------------

def build_per_signal_matrices(chembl_ids, verbose=True):
    """Fetch data and compute per-signal similarity matrices."""
    n = len(chembl_ids)

    if verbose:
        print(f"  Fetching {n} drugs (parallel)...")
    act_by_drug = {}
    moa_map = {}
    mol_map = {}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {}
        for cid in chembl_ids:
            futs[ex.submit(fetch_activities, cid)] = ("act", cid)
            futs[ex.submit(fetch_mechanism, cid)] = ("moa", cid)
            futs[ex.submit(fetch_molecule, cid)] = ("mol", cid)
        for fut in as_completed(futs):
            kind, cid = futs[fut]
            try:
                res = fut.result()
            except Exception:
                res = None
            if kind == "act" and res is not None and not res.empty:
                act_by_drug[cid] = res
            elif kind == "moa" and res:
                moa_map[cid] = res
            elif kind == "mol" and res:
                mol_map[cid] = res

    # Features per drug
    smiles_map = {cid: mol_map.get(cid, {}).get("smiles", "") for cid in chembl_ids}
    atc_map = {cid: mol_map.get(cid, {}).get("atc_codes", []) for cid in chembl_ids}
    moa_text = {cid: moa_map.get(cid, {}).get("moa", "") for cid in chembl_ids}
    scaffold_map = {cid: murcko(smiles_map.get(cid, "")) for cid in chembl_ids}
    assay_descs = {cid: [] for cid in chembl_ids}

    if act_by_drug:
        combined = pd.concat(act_by_drug.values(), ignore_index=True)
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
        for cid, grp in combined.groupby("chembl_id"):
            assay_descs[cid] = grp["assay_description"].dropna().tolist()[:20]
    else:
        combined = pd.DataFrame()

    # Data availability
    n_with_targets = combined["chembl_id"].nunique() if not combined.empty else 0
    n_with_atc    = sum(1 for cid in chembl_ids if atc_map.get(cid))
    n_with_moa    = sum(1 for cid in chembl_ids if moa_text.get(cid))
    n_with_smiles = sum(1 for cid in chembl_ids if smiles_map.get(cid))

    if verbose:
        print(f"  Data availability:")
        print(f"    targets:  {n_with_targets}/{n}")
        print(f"    ATC:      {n_with_atc}/{n}")
        print(f"    MoA:      {n_with_moa}/{n}")
        print(f"    SMILES:   {n_with_smiles}/{n}")

    # Per-signal matrices
    if not combined.empty:
        target_ids = sorted(combined["target_chembl_id"].unique())
        binary = pd.DataFrame(0, index=chembl_ids, columns=target_ids)
        for _, r in combined.iterrows():
            binary.loc[r["chembl_id"], r["target_chembl_id"]] = 1
        median_aff = combined.groupby(
            ["chembl_id", "target_chembl_id"]
        )["value_nM"].median().to_dict()
    else:
        binary = pd.DataFrame()
        target_ids = []
        median_aff = {}

    def target_sim(d1, d2):
        if not target_ids: return 0.0
        a, b = binary.loc[d1].values, binary.loc[d2].values
        return jaccard(set(binary.columns[(a == 1)]), set(binary.columns[(b == 1)]))

    def affinity_sim(d1, d2):
        if not target_ids: return 0.0
        a, b = binary.loc[d1].values, binary.loc[d2].values
        shared = np.array(target_ids)[(a == 1) & (b == 1)]
        if shared.size == 0: return 0.0
        vals = []
        for t in shared:
            a1, a2 = median_aff.get((d1, t)), median_aff.get((d2, t))
            if pd.notna(a1) and pd.notna(a2) and a1 > 0 and a2 > 0:
                vals.append(np.exp(-abs(np.log10(a1) - np.log10(a2))))
        return float(np.mean(vals)) if vals else 0.0

    # Build each signal matrix
    matrices = {}
    for signal_name, fn in [
        ("target",    target_sim),
        ("affinity",  affinity_sim),
        ("structure", lambda a, b: tanimoto(smiles_map.get(a, ""), smiles_map.get(b, ""))),
        ("scaffold",  lambda a, b: 1.0 if (scaffold_map.get(a) and scaffold_map.get(a) == scaffold_map.get(b)) else 0.0),
        ("moa",       lambda a, b: moa_similarity(moa_text.get(a, ""), moa_text.get(b, ""))),
        ("atc",       lambda a, b: atc_sim(atc_map.get(a, []), atc_map.get(b, []))),
        ("assay",     lambda a, b: assay_similarity(assay_descs.get(a, []), assay_descs.get(b, []))),
    ]:
        mat = pd.DataFrame(np.zeros((n, n)), index=chembl_ids, columns=chembl_ids)
        for i, d1 in enumerate(chembl_ids):
            for j, d2 in enumerate(chembl_ids):
                mat.iloc[i, j] = 1.0 if d1 == d2 else fn(d1, d2)
        matrices[signal_name] = mat

    return matrices


def build_correlation_matrix(chembl_ids, verbose=True):
    """Main entry point — entropy-weighted composite correlation matrix."""
    if len(chembl_ids) < 2:
        return pd.DataFrame()

    matrices = build_per_signal_matrices(chembl_ids, verbose=verbose)

    if verbose:
        print(f"\n  Entropy-weighted signal combination:")

    weights = compute_weights(matrices, verbose=verbose)

    # Composite
    composite = sum(weights[name] * matrices[name] for name in matrices)

    if verbose:
        print(f"\n  ✓ Composite matrix built (entropy-weighted)")

    return composite
