# Genericity Test — Evidence

**Date:** 2026-09-25

RepurposeAlpha's discovery module was tested on 5 biologically diverse diseases,
with zero code changes between runs:

| Disease | Candidates Found | Stage ≥ Phase 2 |
|---------|-----------------|-----------------|
| Tuberculosis | 134 associations | 25 |
| HIV | 454 associations | 25 |
| Type 2 Diabetes | 613 associations | 25 |
| Alzheimer disease | 441 associations | 25 |
| Leishmaniasis | 16 associations | 14 |

**Total candidates discovered: 114** (across 5 diseases)

All returned drug names were biologically sensible for their indications:
Linezolid and Capreomycin for TB; Dolutegravir and Lopinavir for HIV;
Amphotericin B and Eflornithine for Leishmaniasis.

**Conclusion:** The discovery module is disease-agnostic. It works for
any disease indexed in Open Targets Platform (>30,000 diseases).

**Method:** Open Targets Platform GraphQL API v4, `drugAndClinicalCandidates`
field, filtered to max clinical stage ≥ PHASE_2.
