# Polymarket Mispricing Detection (CMSE 492 Final Project)

Detect mispriced Polymarket YES/NO markets from snapshot-based price history and order-flow features. Mispricing label: `abs(final_price - last_price) >= 0.15`. Best model: tuned XGBoost (ROC-AUC ≈ 0.94; F1 ≈ 0.62 at threshold 0.50). This README gives new users the rationale, repo layout, and exact run order.

## Table of Contents
- [Overview](#overview)
- [Repo Structure](#repo-structure)
- [Setup](#setup)
- [Run Order](#run-order)
- [Data](#data)
- [Models & Metrics](#models--metrics)
- [Notebook Workflow](#notebook-workflow)
- [Report](#report)
- [Logic & Design Notes](#logic--design-notes)
- [Citation & Disclaimer](#citation--disclaimer)

## Overview
- Goal: flag markets whose last observed price is ≥15 cents away from the eventual final price.
- Approach: supervised classifiers on snapshot features built from price history, volatility, liquidity, and order-flow windows (5/15/60 minutes), plus time-to-resolution and lifecycle features.
- Outputs: tuned models, threshold sweeps, diagnostic figures, and a simple optimistic PnL backtest.

## Repo Structure
```
.
├── requirements.txt
├── data/
│   └── processed/
│       ├── training_data.csv                # raw features (if rebuilt)
│       ├── training_data_clean.csv          # cleaned modeling set
│       ├── model_results_simple_misprice.json
│       ├── model_results_optimized_misprice.json
│       └── threshold_sweep_xgb_misprice.csv
├── models/                                  # saved baseline + tuned models
├── figures/                                 # ROC, PR, confusion, SHAP, calibration, backtest, etc.
├── notebooks/
│   └── model_diagnostics_and_backtest.ipynb
├── report/
│   ├── final_report.tex
│   ├── references.bib
│   └── final_report.pdf
└── src/
    ├── features.py
    ├── clean_data.py
    ├── explore_data.py
    ├── model_simple.py
    ├── model_optimized.py
    └── threshold_tuning.py
```

## Setup
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Order
1) (Optional) Rebuild data from Polymarket APIs  
   ```bash
   python -m src.features       # fetch markets/history/trades → data/processed/training_data.csv
   python -m src.clean_data     # clean/impute → data/processed/training_data_clean.csv
   python -m src.explore_data   # EDA + correlation heatmap → data/processed/
   ```
   Skip if using the provided `data/processed/training_data_clean.csv`.

2) Train baseline models  
   ```bash
   python -m src.model_simple
   ```
   Outputs: `data/processed/model_results_simple_misprice.json`, baseline `models/*_misprice.joblib`.

3) Train optimized (tuned) models  
   ```bash
   python -m src.model_optimized
   ```
   Outputs: `data/processed/model_results_optimized_misprice.json`, tuned `models/*_optimized_misprice.joblib` (logreg, RF, XGB).

4) Threshold sweep on optimized XGBoost  
   ```bash
   python -m src.threshold_tuning
   ```
   Outputs: `data/processed/threshold_sweep_xgb_misprice.csv`, printed best thresholds (F1 ≈ 0.50; precision ≈ 0.70).

5) EDA and training diagnostics (plots, learning curves, runtimes, metrics)  
   ```bash
   python -m src.eda_and_training_diagnostics
   ```
   Generates updated figures and CSV summaries in `figures/` and `data/processed/` for the report.

6) Diagnostics & backtesting notebook  
   ```bash
   jupyter lab notebooks/model_diagnostics_and_backtest.ipynb
   # or
   jupyter notebook notebooks/model_diagnostics_and_backtest.ipynb
   ```
   “Run All” to regenerate diagnostics, SHAP, and PnL/backtest plots (saved to `figures/`).

7) (Optional) Build the report PDF  
   From `report/`:
   ```bash
   pdflatex final_report.tex
   bibtex final_report
   pdflatex final_report.tex
   pdflatex final_report.tex
   ```

## Data
- Clean set: `data/processed/training_data_clean.csv` (7,558 rows, 34 columns; mispricing positives ≈3.7%).
- Features: last/final price, moving averages, volatility/trend/range, time-to-resolution and lifecycle fractions, bid/ask-derived mid, trade counts and buy/sell volumes over 5/15/60 minutes, net/order-flow ratios, snapshot offset hours.
- Target: `y_misprice = 1` if `abs(final_price - last_price) >= 0.15`, else 0.

## Models & Metrics
- Baseline: logistic regression, random forest, XGBoost (`src.model_simple.py`).
- Tuned: randomized search for logreg/RF/XGB (`src.model_optimized.py`).
- Key tuned XGBoost metrics (held-out): ROC-AUC 0.9428; F1 @ 0.50 = 0.6195 (precision 0.6140, recall 0.6250); precision-oriented @ 0.70 = precision 0.6383, recall 0.5357, F1 0.5825.

## Notebook Workflow
`notebooks/model_diagnostics_and_backtest.ipynb`:
- Rebuilds `y_misprice` from `../data/processed/training_data_clean.csv`.
- Loads `../models/xgboost_optimized_misprice.joblib`.
- Evaluates ROC/PR, confusion matrices at 0.50 and 0.70, calibration.
- Plots feature importances and SHAP (summary + dependence).
- Runs an optimistic PnL backtest for thresholds 0.50 and 0.70.
- Saves all figures to `../figures/`.

## Report
- LaTeX: `report/final_report.tex` with `report/references.bib`.
- Compiled PDF: `report/final_report.pdf`.

## Logic & Design Notes
- Snapshot framing: features are computed at fixed offsets before resolution (no sequence model), balancing simplicity and signal from recent price/action windows.
- Class imbalance: strong skew toward non-mispriced; models use class weighting and threshold tuning.
- Interpretability: SHAP and feature importances highlight which liquidity/order-flow and price-trend signals drive predictions.
- Backtest: optimistic, ignores slippage/fees/impact; thresholds trade off recall (0.50) vs precision (0.70).

## Citation & Disclaimer
- Course: CMSE 492, Michigan State University.
- Data: public Polymarket APIs (Gamma, CLOB price history, trades).
- Stack: pandas, numpy, scikit-learn, XGBoost, seaborn, matplotlib, shap, requests, joblib.
- For academic use only; not financial advice.
