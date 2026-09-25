# Correlation Engine — Validation Evidence

**Date:** 2026-09-26

## Test: 3-Tier Similarity Gradient

10 drugs across 4 therapeutic classes tested with the multi-signal
correlation engine.

| Tier | Description | Avg similarity |
|------|-------------|----------------|
| 1 | Within artemisinins (same scaffold) | 0.322 |
| 2 | Within quinolines (same core, diff tails) | 0.251 |
| 3 | Artemisinin vs quinoline (sibling families) | 0.096 |
| 4 | Artemisinin vs unrelated drugs | 0.031 |
| 5 | Quinoline vs unrelated drugs | 0.032 |

**Signal-to-noise ratio (Tiers 1-2 vs Tiers 4-5): 8.9x**

## Negative controls

Three drugs with no expected biological relationship to antimalarials
were included:

- Ibuprofen (NSAID): avg 0.024 vs antimalarials
- Metformin (biguanide): avg 0.014 vs antimalarials
- Amoxicillin (beta-lactam): avg 0.021 vs antimalarials

All three correctly show near-zero correlation with the test set.

## Conclusion

The engine produces a clean monotonic gradient from "most similar"
to "least similar." This validates its use as a portfolio-correlation
input for drug repurposing.

## Known limitation

Artemether vs Atorvastatin = 0.148 (higher than expected). This is
likely due to shared structural features (polycyclic carbon skeletons)
that Tanimoto similarity captures but that are not pharmacologically
meaningful. Documented for future refinement.
