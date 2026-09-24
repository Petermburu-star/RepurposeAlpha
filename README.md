

---

## Phase 2 - Progress

**Vertical:** Prader-Willi Syndrome (PWS)

**Candidates analyzed (4):**
- Carbetocin (OXTR agonist)
- Oxytocin (OXTR agonist)
- Setmelanotide (MC4R agonist)
- Tirzepatide (GLP1R/GIPR dual agonist)

**Key finding:** Carbetocin vs Oxytocin correlation = **0.748**
- Same oxytocin/vasopressin receptor family, ~5x affinity difference
- Markowitz optimizer confirms: moderate correlation is a diversification trap

**Deliverables:**
- data/processed/pws_correlation_matrix.csv
- data/processed/pws_optimal_weights.csv
- data/processed/pws_sensitivity.csv
- reports/pws_correlation_matrix.png
- reports/pws_efficient_frontier.png
- reports/pws_sensitivity_analysis.png

**Methodology:** ChEMBL bioactivity data -> target overlap (Jaccard) + affinity similarity
(log-scale) -> composite correlation -> Markowitz optimization via PyPortfolioOpt.

---

## Phase 3 - Interactive Demo

Run the app locally:

    cd RepurposeAlpha
    streamlit run app/app.py

Then open http://localhost:8501 in your browser.

Features:
- Correlation matrix heatmap (interactive)
- Efficient frontier with Monte Carlo backdrop
- Max Sharpe + Min Volatility portfolios
- Live sensitivity analysis on the correlation assumption
