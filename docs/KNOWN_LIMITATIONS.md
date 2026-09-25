# Known Limitations

*This document is part of RepurposeAlpha's commitment to honest reporting.*

## 1. P(success) prediction from RepoDB is not feasible

### The problem

RepoDB appears to be a suitable supervised-learning dataset for predicting
drug-repurposing success (6,677 approved rows vs 4,123 failed rows). It is not.

### Why

RepoDB is constructed by concatenating two heterogeneous sources:

1. **FDA drug labels** — each drug paired with its primary approved indication.
2. **ClinicalTrials.gov failures** — each drug paired with a failed repurposing attempt.

These sources produce structurally different data. A given disease usually appears
in only one of the two sources — either as a label indication or as a trial target.
As a result:

- 51.7% of diseases appear with only successful outcomes
- 40.1% of diseases appear with only failed outcomes
- Only 8.2% of diseases have mixed outcomes

This means the label is recoverable from disease identity alone — a model
trained on this data achieves AUC > 0.98 through data leakage, not biological
signal. Group-aware cross-validation (splitting by drug) confirms the leak
is structural, not memorization-based.

### What this means for RepurposeAlpha

RepurposeAlpha **does not claim** to predict drug-repurposing success with
machine learning. It uses **portfolio construction** (Modern Portfolio Theory)
on biological correlation data — a completely different and defensible method.

The `ml_model_metrics.json` file in `data/processed/` documents the leaky
results for transparency. The leaky model is not used in the product.

## 2. Correlation is computed from public bioactivity data

The correlation matrix between candidates is derived from ChEMBL bioactivity
records. It reflects *measured binding similarity*, not clinical outcome
similarity. Two drugs with identical binding profiles may still behave
differently in patients.

## 3. Cost-per-phase figures are literature estimates

The failure-cost model uses standard industry estimates (DiMasi 2016,
Wouters 2020). These are widely accepted but not derived from RepoDB.
Cite the source before using for investor decisions.

## 4. RepoDB's internal base rate is not the real-world base rate

RepoDB is ~62% approved. Real-world repurposing success rates are closer
to 10–30%. RepurposeAlpha does not rely on RepoDB's base rate for any
portfolio decision.

## 5. Single-disease scope

The current prototype focuses on Prader-Willi Syndrome. Generalization to
other rare diseases requires re-running the pipeline for each vertical,
which the architecture supports but has not been validated at scale.

---

*If you spot an issue we haven't documented, please open an issue on GitHub.*