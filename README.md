# Classify Correctly — Card-Not-Present Fraud Risk Detector

> **Razorpay AI Buildathon · AI Risk Manager track**  
> Strictly defensive: scores risk, never generates or facilitates fraud.

---

## Problem Statement

Card-not-present (CNP) fraud is the dominant loss vector for online payment processors. A transaction arrives with no physical card verification — the system must decide in milliseconds whether to clear or flag it. The challenge is extreme class imbalance: genuine fraud is rare (~0.17% of transactions), so a model that blindly clears everything achieves 99.83% accuracy while missing all fraud. **Accuracy is not the metric. PR-AUC and total expected cost are.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Transaction Input (dict)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FraudScorer (src/scorer.py)                     │
│                                                             │
│  1. Feature validation & default-fill                       │
│  2. XGBoost predict_proba → risk score [0.0, 1.0]          │
│  3. score >= threshold → FLAGGED / CLEARED                  │
│  4. SHAP TreeExplainer → top-3 contributing features        │
│  5. Audit log append (logs/audit_trail.jsonl)               │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┴───────────────┐
          ▼                              ▼
   ScoringResult dict             Audit Trail
   {decision, confidence,         (append-only JSONL,
    top_reasons, timestamp,        never silent-drops)
    model_version, input_hash}
```

---

## Dataset

**Kaggle Credit Card Fraud Detection** (`mlg-ulb/creditcardfraud`)

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraud transactions | 492 (0.172%) |
| Features | 28 PCA-anonymised (V1–V28) + Amount + Time |
| Labels | Binary: 0 = legitimate, 1 = fraud |

**Split (stratified, leak-free):**

| Split | Rows | Fraud | Fraud % |
|---|---|---|---|
| Train | ~170,884 | ~295 | 0.173% |
| Val | ~56,962 | ~98 | 0.172% |
| Test | ~56,961 | ~99 | 0.174% |

> The test set was isolated immediately after splitting and was **never touched** during training, validation, or threshold tuning. It was accessed exactly once — for the final metrics report.

---

## Models

### Baseline: Logistic Regression
- `class_weight='balanced'` to handle imbalance
- StandardScaler on Amount + Time (V1–V28 are already PCA-normalised)
- Threshold selected on validation set via cost-optimal sweep

### Primary: XGBoost
- `scale_pos_weight = n_negatives / n_positives` (~579) — built-in imbalance handling
- Hyperparameter tuning: 40 Optuna trials, objective = PR-AUC on validation set
- Threshold selected on validation set (never the test set)

### Threshold Selection Policy

Thresholds are **not** set at 0.5. We sweep [0.01, 0.99] on the validation set and choose:

```
threshold* = argmin_t [ FP(t) × 150 + FN(t) × avg_fraud_amount ]
```

`avg_fraud_amount` is computed from **training data only**. This makes the threshold choice explicit, auditable, and tied to real business cost.

A **recall ≥ 0.85 reference point** is also reported for comparison.

---

## Results — Held-Out Test Set

> ⚠️ These numbers come from the test set the model never saw. Accuracy is omitted — it is near-meaningless at 0.17% fraud rate.
> Full numbers: [`reports/metrics.json`](reports/metrics.json)

### XGBoost (Primary)

| Metric | Value |
|---|---|
| **PR-AUC** | **0.8753** |
| **ROC-AUC** | 0.9724 |
| **Precision** | **0.9091** |
| **Recall** | **0.8163** |
| **F1** | **0.8602** |
| Threshold (cost-optimal) | 0.8879 |
| TP / FP / FN / TN | 80 / 8 / 18 / 56,856 |

### Logistic Regression (Baseline)

| Metric | Value |
|---|---|
| PR-AUC | 0.7209 |
| ROC-AUC | 0.9726 |
| Precision | 0.5563 |
| Recall | 0.8571 |
| F1 | 0.6747 |
| Threshold (cost-optimal) | 0.9900 |
| TP / FP / FN / TN | 84 / 67 / 14 / 56,797 |

> **Why XGBoost wins:** LR achieves slightly higher recall (0.857 vs 0.816) but at the cost of 8× more false positives (67 vs 8). The cost-optimal threshold policy correctly penalises this — XGBoost's total expected cost is **₹3,543 vs ₹11,872** for LR, a 70% cost reduction.

> **Recall ≥ 0.85 reference (XGBoost, val set):** At threshold 0.3259, precision = 0.3774, recall = 0.8687. This point catches more fraud but generates ~8× more false alarms — unacceptable for a high-volume processor at scale.

---

## False-Positive Cost

**₹150 per false positive** is estimated at approximately 10 minutes of a support agent's time at a blended hourly cost of ₹900 (₹15/min × 10 min). This covers agent time to review the flagged transaction plus customer friction (payment hold, support call, potential churn).

| Metric | XGBoost | LR Baseline |
|---|---|---|
| FP count (test set) | **8** | 67 |
| FP per 1,000 transactions | **0.14** | 1.18 |
| Total FP cost | **₹1,200** | ₹10,050 |
| FN count (test set) | 18 | 14 |
| Total FN cost (FN × ₹130.15 avg fraud) | ₹2,343 | ₹1,822 |
| **Total expected cost** | **₹3,543** | ₹11,872 |

XGBoost's cost-optimal threshold (0.8879) produces **8 false positives and 18 false negatives** on 56,962 test transactions — a total expected cost of **₹3,543**. LR's cost-optimal threshold misses fewer frauds (14 FN) but drowns reviewers in 67 false alarms, costing 3.4× more overall.

---

## Explainability

Every decision comes with SHAP-derived reasons. Example:

```json
{
  "decision": "FLAGGED",
  "confidence": 0.934,
  "threshold": 0.42,
  "top_reasons": [
    "Behavioural pattern 4 increases fraud risk (SHAP = +1.24)",
    "Behavioural pattern 14 increases fraud risk (SHAP = +0.87)",
    "Transaction amount decreases fraud risk (SHAP = -0.31)"
  ],
  "timestamp": "2026-09-05T06:00:00+00:00",
  "model_version": "xgb-v1.0",
  "input_hash": "a3f9b2c1"
}
```

V1–V28 are labelled "Behavioural pattern N" — honest about the PCA anonymisation while still meaningful to a human reviewer.

---

## Audit Trail

Every scored transaction appends to `logs/audit_trail.jsonl` (append-only, never silent-drops):

```json
{
  "input_hash": "a3f9b2c1",
  "timestamp": "2026-09-05T06:00:00+00:00",
  "decision": "FLAGGED",
  "confidence": 0.934,
  "threshold": 0.42,
  "top_reasons": ["..."],
  "model_version": "xgb-v1.0",
  "error": null,
  "amount": 149.62
}
```

---

## Known Limitations

1. **PCA anonymisation prevents feature engineering.** V1–V28 are opaque — we cannot add domain-specific features (merchant category, geolocation, velocity) that would materially improve recall.

2. **Static training data.** No concept drift handling. A production system needs periodic retraining as fraud patterns evolve.

3. **No velocity/sequence features.** CNP fraud often involves multi-transaction patterns. This dataset treats each transaction independently.

4. **Dataset represents European cardholders (2013).** Distribution shift may be significant for Indian payment patterns.

5. **SHAP explanations for V-features are opaque.** Honest, but less useful to a fraud analyst than named features.

6. **Threshold was tuned on in-distribution validation data.** In deployment, the optimal threshold may shift as the fraud mix changes.

---

## What We Would Do With More Time

- **Sequence/velocity features:** Rolling transaction counts per card per hour — likely the single biggest recall improvement
- **Isotonic calibration:** Ensure confidence scores are well-calibrated probabilities
- **Concept drift monitoring:** Track PR-AUC on a sliding window; alert on degradation
- **Merchant-aware features:** Category-specific fraud rates shift the base rate substantially
- **Online learning:** Partial-fit on confirmed fraud labels from chargeback feedback

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download & split data (requires ~/.kaggle/kaggle.json)
python src/data_prep.py

# Train models
python src/train.py

# Evaluate on held-out test set (run exactly once)
python src/evaluate.py

# Score a batch via CLI
python cli.py --input data/processed/test.csv --limit 50

# Launch Streamlit demo
streamlit run app.py

# Run unit tests
python -m pytest tests/ -v
```

---

## Stack

| Component | Library |
|---|---|
| Data | pandas, numpy |
| Models | scikit-learn, XGBoost |
| Tuning | Optuna |
| Explainability | SHAP |
| Demo | Streamlit |
| Plots | matplotlib, seaborn |

---

## Defensive-Only Statement

This system is a **risk scorer only**. It accepts transaction features, outputs a binary decision + confidence + reasons, and logs every call to an append-only audit trail. It does not and cannot generate synthetic transactions, suggest evasion strategies, produce card credentials, or act as an attack proxy.
