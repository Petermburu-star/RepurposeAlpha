# Phase 1 — Discovery Report
**Generated:** 2026-09-24 16:18
**Project:** RepurposeAlpha

## 1. Dataset Summary
- Source: RepoDB (Figshare file 7341422, KG-Hub official distribution)
- Rows: 10,800
- Columns: 7
- File: data/raw/repodb_full.csv (1,029,743 bytes)

## 2. Schema

| Role | Column | Unique | Nulls |
|------|--------|--------|-------|
| Drug name | drug_name | 1,571 | 260 |
| Drug ID | drug_id | 1,571 | 0 |
| Indication name | ind_name | 2,051 | 0 |
| Indication ID | ind_id | 2,051 | 0 |
| Status | status | 4 | 0 |
| Phase | phase | 6 | 6,677 |
| Detailed status | DetailedStatus | 752 | 7,612 |

## 3. Outcome Distribution

status
Approved      6677
Terminated    2877
Withdrawn      662
Suspended      584

## 4. KEY STRUCTURAL FINDING

RepoDB is a union of two datasets:

1. 6,677 already-approved drug-indication pairs (61.8%)
   Positive class. All have phase = NaN.
2. 4,123 failed trial attempts (38.2%)
   Terminated, Suspended, or Withdrawn. All carry a phase value.

Implication: The failure phase distribution gives us an empirical
cost-of-failure model, the primary input to our Markowitz rNPV optimizer.

## 5. Empirical Cost Model

                 n_failures    mean_cost   total_cost  pct_of_total
phase                                                              
Phase 3                 714  100000000.0  71400000000          50.6
Phase 2                2076   25000000.0  51900000000          36.8
Phase 1                 789   10000000.0   7890000000           5.6
Phase 1/Phase 2         382   15000000.0   5730000000           4.1
Phase 2/Phase 3         105   40000000.0   4200000000           3.0
Phase 0                  57    1000000.0     57000000           0.0

- Total historical capital lost: $141,177,000,000
- Average cost per failed attempt: $34,241,329
- Phase 2 + Phase 3 = 87.4% of all capital lost

## 6. Rare-Disease Opportunity
- Unique indications: 2,051
- Indications with 1-3 candidates: 1,285
- Shortlist: data/processed/rare_disease_shortlist.csv

## 7. Data Quality Notes
- drug_name missing in 260 rows - use drug_id as join key.
- phase missing in 6,677 rows - all correspond to approved pairs.
- DetailedStatus free-text (752 values) - reserved for NLP.
- Imbalance: {'Approved': 0.618, 'Terminated': 0.266, 'Withdrawn': 0.061, 'Suspended': 0.054}
- Caveat: Cost-per-phase figures are literature estimates (DiMasi 2016,
  Wouters 2020) and must be cited before investor use.

## 8. Next Actions (Phase 2)
1. Select first disease vertical from shortlist.
2. Join drug_id (DrugBank) to ChEMBL for bioactivity + target data.
3. Build rNPV scoring function calibrated on empirical phase-failure costs.
4. Run first Markowitz portfolio optimization.

## 9. Security Checklist
- [x] .env excluded via .gitignore
- [x] SECURITY.md committed
- [ ] Encryption layer (Phase 2)
- [ ] RBAC + 2FA (Phase 3)
- [ ] Penetration test (Phase 4)
