# RepurposeAlpha

**Portfolio optimization for drug repurposing.**

Treating candidate drugs as correlated, risky assets — and allocating
validation resources the way a hedge fund allocates capital.

---

## The Problem

Biotech loses an estimated **$141 billion per decade** to failed drug
repurposing attempts. Over **50% of that loss** happens at Phase 3 —
the most expensive stage. Most of these failures are predictable from
public data. The industry has no systematic way to spot them first.

## The Insight

A biotech's pipeline of repurposing candidates has the same structure
as an investment portfolio: assets, returns, volatilities, correlations.
Modern Portfolio Theory can tell scientists **which candidates to validate**,
**in what sequence**, and **at what level of investment** — maximizing
the probability of success while minimizing capital at risk.

## The PWS Case Study

Applied to Prader-Willi Syndrome, our model finds that the two leading
oxytocin-pathway candidates (Carbetocin, Oxytocin) are **75% correlated** —
essentially the same bet. The 2026 COMPASS Phase 3 failure of Carbetocin
confirmed this concentration risk; our tool would have flagged it in 2022.

Full write-up: [`docs/case_study_pws.md`](docs/case_study_pws.md)

## Quick Start

```bash
git clone https://github.com/Petermburu-star/RepurposeAlpha.git
cd RepurposeAlpha
pip install -r requirements.txt
streamlit run app/app.py
```

Then open http://localhost:8501 and log in.

## Features

- **Correlation matrix** — biological similarity between candidate drugs
- **Efficient frontier** — Monte Carlo + Markowitz optimization
- **Optimal portfolio** — Max Sharpe + Min Volatility allocations
- **Live sensitivity** — how allocations change as correlation varies
- **Authentication** — bcrypt-hashed passwords, session management
- **RBAC** — admin / analyst / viewer roles
- **Encryption at rest** — Fernet (AES-128 + HMAC-SHA256)
- **Audit logging** — every login and action recorded

## Architecture

```
RepurposeAlpha/
├── app/                  Streamlit application
│   ├── app.py           Main UI (role-aware tabs)
│   ├── auth.py          Authentication + RBAC
│   └── secure_db.py     Encrypted SQLite wrapper
├── data/
│   ├── raw/             RepoDB, ChEMBL (gitignored)
│   └── processed/       Correlation matrices, weights
├── docs/
│   └── case_study_pws.md  Investor case study
├── notebooks/           Analysis notebooks
├── reports/             Generated figures
└── tests/               Test suite
```

## Stack

- **Data**: RepoDB, ChEMBL, ClinicalTrials.gov (all free, public)
- **Modeling**: pandas, NumPy, scikit-learn, PyPortfolioOpt
- **UI**: Streamlit, Plotly
- **Security**: bcrypt, cryptography (Fernet)
- **Cost**: $0 — every dependency is free and open-source

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Data pipeline + empirical cost model | ✅ |
| 2 | PWS vertical + Markowitz optimizer | ✅ |
| 3 | Interactive app + enterprise security | ✅ |
| 4 | ML model for P(success) from RepoDB | 🔜 |
| 5 | Budget-constrained multi-disease solver | Planned |

## Security

See [`SECURITY.md`](SECURITY.md) for our commitments and disclosure policy.

## Author

**Peter Mburu Kariuki** — [pmburu346@gmail.com](mailto:pmburu346@gmail.com)

---

*RepurposeAlpha is a Phase 3 prototype. All computations are deterministic,
auditable, and reproducible. Data sources are public; no proprietary
information is used.*

## What RepurposeAlpha Is (and Isn't)

**Is:** A portfolio construction tool that applies Modern Portfolio Theory
to drug repurposing. It identifies correlated candidates and recommends
allocations that maximize expected risk-adjusted return.

**Isn't:** A machine learning predictor of clinical success. RepoDB's
structure makes it unsuitable for supervised learning (documented in
[`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md)).

We could have shipped a model claiming AUC = 0.98. We didn't, because
that model would have been wrong for subtle, structural reasons that
would destroy credibility in a technical due-diligence review. Honesty
compounds; hype does not.

---

## The Seven Features

RepurposeAlpha answers eight questions about any drug repurposing candidate:

| Question | Feature | Tab |
|---|---|---|
| What should I test? | Markowitz portfolio optimization | Portfolio |
| When should I kill it? | Real Options valuation | Real Options |
| Can I protect it commercially? | Freedom-to-Operate screening | FTO |
| How should I run the trial? | Adaptive trial design | Trials |
| What else does it treat? | Cross-disease mechanism discovery | Cross-Disease |
| Has it failed before? | Negative results intelligence | Prior Results |
| What pairs well with it? | Combination synergy predictor | Synergy |
| What price is sustainable? | Cost-based pricing framework | Pricing |

Every numeric input is sourced. Every result is reproducible.

## What Makes This Different

Existing tools serve the commercial pipeline. They optimize for big pharma.

RepurposeAlpha is built for non-commercial repurposing — academic labs,
rare disease foundations, neglected tropical disease researchers — where
market forces have failed but scientific need is greatest.

## Release History

- **v0.1.0** (2026-09-25) — Level 1 complete: discovery, correlation, registry, PDF
- **v0.2.0** (2026-09-28) — Seven-feature release
