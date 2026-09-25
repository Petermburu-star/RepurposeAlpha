# RepurposeAlpha — Level 1 Acceptance Criteria

**Definition:** A functionally real tool that works end-to-end for any disease,
with every assumption traceable to a source, and exports a defensible report.

**Target completion:** 6 months from project start

---

## The 8 Acceptance Criteria

Each criterion is binary. Either it's done or it isn't.

### AC-1: Generic Disease Input
- [ ] User can type or select ANY disease from a dropdown
- [ ] Tool returns candidate drugs automatically from public data
- [ ] No hardcoded disease in the codebase
- [ ] Works for at least 3 different diseases (PWS, malaria, +1)

### AC-2: Automated Correlation Engine
- [ ] Given candidates, computes target overlap automatically
- [ ] Given targets, computes affinity similarity automatically
- [ ] Produces a correlation matrix without manual input
- [ ] Handles the edge case of a single candidate gracefully

### AC-3: Assumption Registry
- [ ] Every numeric input in the UI has a `source` field
- [ ] Every input has a `confidence` label (high/medium/low)
- [ ] Live data used where available (risk-free rate, ChEMBL bioactivity)
- [ ] Users can override any input, and overrides are logged

### AC-4: Markowitz Optimizer
- [ ] Produces Max Sharpe + Min Volatility portfolios
- [ ] Handles correlated assets correctly (already verified)
- [ ] Reports expected return, volatility, and Sharpe ratio
- [ ] Works for candidate sets from 1 to 50 drugs

### AC-5: Sensitivity Analysis
- [ ] User can vary any assumption and see the portfolio change
- [ ] Works for correlations, returns, volatilities
- [ ] Charts update in real-time

### AC-6: One-Click PDF Export
- [ ] Generates a report with: correlation matrix, efficient frontier,
  portfolio weights, sensitivity analysis, and assumptions table with sources
- [ ] Includes project branding and date
- [ ] Usable as-is in a pipeline review meeting

### AC-7: Security (Already Complete)
- [x] bcrypt authentication
- [x] RBAC (admin/analyst/viewer)
- [x] Fernet encryption at rest
- [x] Audit logging

### AC-8: Documentation
- [ ] README explains how to run the tool
- [ ] KNOWN_LIMITATIONS.md is current
- [ ] METHODOLOGY.md explains every computational step
- [ ] At least one case study (PWS) is complete

---

## What Level 1 Is NOT

- NOT enterprise-sellable (that's Level 2 — needs SOC 2, pentest, SSO)
- NOT a predictor of clinical success (RepoDB can't support this)
- NOT a substitute for wet-lab validation (it prioritizes, not proves)
- NOT a multi-user SaaS (single-tenant is fine at Level 1)

## Progress Tracking

| Criterion | Status | Notes |
|-----------|--------|-------|
| AC-1 | 🔜 | The next 4-6 weeks of work |
| AC-2 | 🟡 | Works for PWS, needs generalization |
| AC-3 | 🔜 | Design pattern established (Cell 58) |
| AC-4 | ✅ | Complete |
| AC-5 | ✅ | Complete |
| AC-6 | 🔜 | Not started |
| AC-7 | ✅ | Complete |
| AC-8 | 🟡 | README done, methodology pending |

---

*Review this file at the start of every session. Update the table.*