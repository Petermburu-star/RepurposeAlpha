# Notebooks

Development notebooks for RepurposeAlpha, in chronological order.

| # | Notebook | What it covers |
|---|---|---|
| 01 | `01_phase1_repodb_discovery.ipynb` | Original data pipeline. RepoDB ingestion, empirical cost model, rare-disease shortlist. |
| 02 | `02_phase4_ml_investigation.ipynb` | ML model attempt on RepoDB and discovery of structural data leakage. |
| 03 | `03_phase5_docs_and_whitepaper.ipynb` | Generation of HTML explainer, technical whitepaper, KEMRI email draft. |
| 04 | `04_phase6_executive_summary.ipynb` | Executive Summary v2, verdict logic, candidate selection. |

## How to use these

These notebooks are **development history**, not the product. The tool lives in:

- `app/` — Streamlit application
- `src/` — Python modules
- `config/` — assumption registry

To run the tool:

```
streamlit run app/app.py
```

## Environment note

Each notebook has a hardcoded `PROJECT_ROOT` path at the top.
Update it if you cloned the repo elsewhere.