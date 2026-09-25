"""
RepurposeAlpha — entropy-based adaptive weighting (v2).

Weights based on each signal's information content, with a variance
floor to exclude constant/uninformative signals.
"""
import numpy as np
import pandas as pd


def signal_stats(sim_matrix: pd.DataFrame) -> dict:
    """Return entropy, variance, and off-diagonal mean of a similarity matrix."""
    if sim_matrix.empty:
        return {"entropy": 1.0, "variance": 0.0, "mean": 0.0}

    n = len(sim_matrix)
    vals = []
    for i in range(n):
        for j in range(n):
            if i != j:
                vals.append(sim_matrix.iloc[i, j])

    if not vals:
        return {"entropy": 1.0, "variance": 0.0, "mean": 0.0}

    vals = np.array(vals)
    variance = float(np.var(vals))
    mean = float(np.mean(vals))

    # Shannon entropy on 10-bin histogram
    hist, _ = np.histogram(vals, bins=10, range=(0, 1), density=False)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    H = -np.sum(probs * np.log(probs))
    H_max = np.log(10)
    entropy = float(H / H_max) if H_max > 0 else 1.0
    # Clamp to [0, 1] to kill -0.0 artifacts
    entropy = max(0.0, min(1.0, entropy))

    return {"entropy": entropy, "variance": variance, "mean": mean}


def compute_weights(similarity_matrices: dict, verbose: bool = True,
                    variance_floor: float = 0.001) -> dict:
    """
    Entropy-based weights with a variance floor.

    A signal is only eligible if its off-diagonal variance exceeds
    the floor. Constant signals (all zeros, all ones) are excluded.
    """
    stats = {}
    for name, mat in similarity_matrices.items():
        stats[name] = signal_stats(mat)

    # Eligibility: exclude constant signals
    eligible = {
        name: s for name, s in stats.items()
        if s["variance"] >= variance_floor
    }

    if not eligible:
        if verbose:
            print("  ⚠️  No signal has sufficient variance — using uniform weights")
        n = len(similarity_matrices)
        return {name: 1.0/n for name in similarity_matrices}

    # Inverse entropy on eligible signals only
    inform = {name: (1.0 - s["entropy"]) for name, s in eligible.items()}
    total = sum(inform.values())

    if total <= 0:
        weights = {name: 1.0/len(eligible) for name in eligible}
    else:
        weights = {name: v / total for name, v in inform.items()}

    # Zero weight for ineligible signals
    for name in similarity_matrices:
        if name not in weights:
            weights[name] = 0.0

    if verbose:
        print("  Entropy analysis (variance floor = {:.4f}):".format(variance_floor))
        print(f"  {'signal':12s}  {'entropy':>8s}  {'variance':>10s}  {'weight':>8s}")
        for name in sorted(similarity_matrices, key=lambda x: -weights[x]):
            s = stats[name]
            tag = "" if s["variance"] >= variance_floor else "  (excluded)"
            print(f"    {name:12s}  {s['entropy']:>8.3f}  {s['variance']:>10.5f}  {weights[name]:>8.3f}{tag}")
        print(f"  Total weight: {sum(weights.values()):.3f}")

    return weights
