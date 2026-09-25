"""
RepurposeAlpha — correlation engine self-validation.

Given a set of drugs with known ATC classifications, this module computes
whether the correlation engine correctly clusters therapeutic families.

Metric: ATC enrichment ratio
  = (avg within-family similarity) / (avg between-family similarity)

Interpretation:
  < 1.5   -> engine is not discriminating (poor)
  1.5-3   -> weak discrimination
  3-10    -> strong discrimination
  > 10    -> very strong (families are cleanly separated)
"""
import sys
import numpy as np
from pathlib import Path
from itertools import combinations

# Bring in correlation fetchers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from correlation import fetch_molecule, build_correlation_matrix


def fetch_atc(chembl_id: str) -> str:
    """Return the primary ATC code for a drug (or '' if none)."""
    mol = fetch_molecule(chembl_id)
    atcs = mol.get("atc_codes") or []
    return atcs[0] if atcs else ""


def atc_family(atc_code: str, level: int = 4) -> str:
    """Extract ATC prefix at a given level (e.g. 'P01B' at level 4)."""
    if not atc_code or len(atc_code) < level:
        return ""
    return atc_code[:level]


def compute_enrichment(chembl_ids: list, atc_level: int = 4,
                        verbose: bool = True) -> dict:
    """Compute within-family vs between-family correlation enrichment."""
    atc_map = {cid: fetch_atc(cid) for cid in chembl_ids}
    families = {}
    for cid, atc in atc_map.items():
        fam = atc_family(atc, atc_level)
        if fam:
            families.setdefault(fam, []).append(cid)

    n_with_atc = sum(len(v) for v in families.values())

    if verbose:
        print(f"  Drugs with ATC: {n_with_atc}/{len(chembl_ids)}")
        print(f"  Distinct families: {len(families)}")
        for fam, ids in families.items():
            print(f"    {fam}: {len(ids)} drugs")

    corr = build_correlation_matrix(chembl_ids, verbose=False)
    if corr.empty:
        return {"error": "empty correlation matrix"}

    within_sims = []
    between_sims = []

    for d1, d2 in combinations(chembl_ids, 2):
        if d1 not in corr.index or d2 not in corr.index:
            continue
        sim = corr.loc[d1, d2]
        fam1 = atc_family(atc_map.get(d1, ""), atc_level)
        fam2 = atc_family(atc_map.get(d2, ""), atc_level)

        if fam1 and fam2:
            if fam1 == fam2:
                within_sims.append(sim)
            else:
                between_sims.append(sim)

    within_avg  = float(np.mean(within_sims))  if within_sims  else 0.0
    between_avg = float(np.mean(between_sims)) if between_sims else 0.0
    enrichment  = (within_avg / between_avg) if between_avg > 0 else float("inf")
    confidence  = min(enrichment / 10.0, 1.0) if np.isfinite(enrichment) else 1.0

    result = {
        "n_drugs":         len(chembl_ids),
        "n_with_atc":      n_with_atc,
        "n_families":      len(families),
        "n_within_pairs":  len(within_sims),
        "n_between_pairs": len(between_sims),
        "within_avg":      within_avg,
        "between_avg":     between_avg,
        "enrichment":      enrichment,
        "confidence":      confidence,
        "per_family":      families,
    }

    if verbose:
        print(f"\n  Within-family pairs:  {len(within_sims)}  (avg sim {within_avg:.3f})")
        print(f"  Between-family pairs: {len(between_sims)} (avg sim {between_avg:.3f})")
        print(f"  Enrichment ratio:     {enrichment:.2f}x")
        print(f"  Confidence score:     {confidence:.2f}")

    return result


def confidence_label(enrichment: float) -> str:
    """Human-readable label for an enrichment ratio."""
    if not np.isfinite(enrichment):
        return "inf (no between-family pairs)"
    if enrichment < 1.5:
        return "poor — engine not discriminating"
    if enrichment < 3:
        return "weak discrimination"
    if enrichment < 10:
        return "strong discrimination"
    return "very strong — families cleanly separated"


# ============================================================
# Negative-control confidence metric (validated 2026-09-26)
# ============================================================
# Instead of relying only on ATC families, this function compares
# "known-related" pairs against "known-unrelated" pairs.
#
# Ratio interpretation (validated against antimalarial test set):
#   < 2    -> poor
#   2-4    -> weak
#   4-8    -> strong
#   > 8    -> very strong

def confidence_from_ratio(ratio: float) -> tuple:
    """Return (label, score) from a within/between ratio."""
    if ratio >= 8.0:
        return ("very strong", 1.0)
    if ratio >= 4.0:
        return ("strong", 0.7)
    if ratio >= 2.0:
        return ("weak", 0.4)
    return ("poor", 0.0)


def two_tier_validation(chembl_ids: list, related_pairs: list,
                         unrelated_pairs: list, verbose: bool = True) -> dict:
    """
    Validate the engine against explicit related/unrelated pairs.

    related_pairs:   [(chembl_id_a, chembl_id_b), ...] expected similar
    unrelated_pairs: [(chembl_id_a, chembl_id_b), ...] expected dissimilar
    """
    corr = build_correlation_matrix(chembl_ids, verbose=False)

    if corr.empty:
        return {"error": "empty matrix"}

    related_sims   = [corr.loc[a, b] for a, b in related_pairs
                      if a in corr.index and b in corr.index]
    unrelated_sims = [corr.loc[a, b] for a, b in unrelated_pairs
                      if a in corr.index and b in corr.index]

    related_avg   = float(np.mean(related_sims))   if related_sims   else 0.0
    unrelated_avg = float(np.mean(unrelated_sims)) if unrelated_sims else 0.0
    ratio = (related_avg / unrelated_avg) if unrelated_avg > 0 else float("inf")
    label, score = confidence_from_ratio(ratio)

    result = {
        "n_related_pairs":   len(related_sims),
        "n_unrelated_pairs": len(unrelated_sims),
        "related_avg":       related_avg,
        "unrelated_avg":     unrelated_avg,
        "ratio":             ratio,
        "label":             label,
        "confidence_score":  score,
    }

    if verbose:
        print(f"  Related pairs:   {len(related_sims)}  (avg {related_avg:.3f})")
        print(f"  Unrelated pairs: {len(unrelated_sims)} (avg {unrelated_avg:.3f})")
        print(f"  Ratio:           {ratio:.2f}x  -> {label} (score {score:.2f})")

    return result
