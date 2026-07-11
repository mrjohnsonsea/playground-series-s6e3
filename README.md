# Predict Customer Churn — Kaggle Playground Series S6E3

Binary classification project predicting the probability that a telecom customer will churn, built for [Kaggle Playground Series — Season 6, Episode 3](https://www.kaggle.com/competitions/playground-series-s6e3/). The competition is scored on **ROC-AUC**.

**Best model:** tuned XGBoost — **0.9162 ± 0.0008 CV ROC-AUC** (5-fold stratified).

---

## Problem

Given ~594k labeled customer records, predict the churn probability for ~255k unlabeled test customers. Each record describes a telecom subscriber across 19 features — demographics (gender, senior citizen, partner, dependents), account tenure and billing (`tenure`, `MonthlyCharges`, `TotalCharges`, contract type, payment method, paperless billing), and subscribed services (phone, internet, and add-ons like online security, backup, device protection, tech support, and streaming).

The dataset is synthetically generated from the classic Telco Customer Churn dataset, so it is clean but large, with a meaningful class imbalance (~23% churn).

## Approach

The full workflow lives in [`notebooks/01_predict_customer_churn.ipynb`](notebooks/01_predict_customer_churn.ipynb):

1. **Data loading** — pulled directly via `kagglehub`; `SeniorCitizen` normalized to Yes/No to match the other binary columns and the `Churn` target mapped to 1/0.
2. **Stratified train/dev split** (80/20) to hold out a clean evaluation set while preserving class balance.
3. **Exploratory analysis** — target balance, numeric distributions (`tenure`, `MonthlyCharges`, `TotalCharges`) split by churn, and per-category churn rates.
4. **Preprocessing** — a `ColumnTransformer` one-hot-encodes categoricals (`handle_unknown='ignore'`) and passes numerics through, wrapped in an sklearn `Pipeline` so preprocessing is fit inside cross-validation and carries through to inference.
5. **Model bake-off** — Logistic Regression, Random Forest, LightGBM, and XGBoost evaluated on the dev set by ROC-AUC.
6. **Cross-validation** of the winning model (5-fold stratified) to confirm the dev score generalizes.
7. **Hyperparameter tuning** with **Optuna** (Bayesian search over XGBoost's tree, sampling, and regularization params, optimizing CV ROC-AUC).
8. **Final fit, feature-importance analysis, and submission** generation.

## Key findings from EDA

- **Contract type** is the strongest signal — month-to-month customers churn at ~42% vs. ~6% (one-year) and ~1% (two-year).
- **Fiber-optic** internet customers churn at ~41%, vs. ~10% for DSL.
- **Electronic-check** payers churn at ~49%, far above the ~7–8% for automatic and mailed-check methods.
- Churn concentrates in **early tenure** and lower `TotalCharges`; customers **without add-on services** and **senior citizens** churn more.

## Results

Dev-set ROC-AUC across candidate models:

| Model | Dev ROC-AUC | Fit time |
|---|---|---|
| **XGBoost** | **0.9168** | 9.4s |
| LightGBM | 0.9165 | 6.3s |
| Random Forest | 0.9126 | 15.9s |
| Logistic Regression | 0.9080 | 1.7s |

XGBoost was selected and cross-validated:

| Metric | 5-fold CV |
|---|---|
| ROC-AUC | 0.9162 ± 0.0008 |
| Accuracy | 0.8610 ± 0.0005 |
| Precision | 0.7118 ± 0.0029 |
| Recall | 0.6437 ± 0.0034 |
| F1 | 0.6760 ± 0.0011 |

The tight cross-validation spread (±0.0008 AUC) confirms a stable model rather than a lucky split, and Optuna tuning matched the well-chosen baseline — evidence the gradient-boosted default was already near the ceiling for this feature set.

## Tech stack

Python 3.14 · pandas · NumPy · scikit-learn · XGBoost · LightGBM · Optuna · Matplotlib / Seaborn · [`uv`](https://github.com/astral-sh/uv) for dependency management · `kagglehub` for data access.

## Reproducing

```bash
# Install dependencies into a virtual environment
uv sync

# Launch Jupyter and run the notebook end-to-end
uv run jupyter lab notebooks/01_predict_customer_churn.ipynb
```

The notebook downloads the competition data via `kagglehub` (a Kaggle account and API credentials are required) and writes `submissions/submission.csv`.

## Project structure

```
.
├── notebooks/
│   └── 01_predict_customer_churn.ipynb   # full analysis & modeling workflow
├── data/                                  # competition CSVs (fetched via kagglehub)
├── submissions/
│   └── submission.csv                     # test-set churn probabilities
├── pyproject.toml                         # dependencies (managed by uv)
└── README.md
```
