# Polymarket Mispricing Detection (CMSE 492 Final Project)

Machine learning pipeline to flag mispriced Polymarket YES/NO markets using snapshot-based price history, volatility, liquidity, and order-flow features. Mispricing label: `abs(final_price - last_price) >= 0.15`. Best performer: tuned XGBoost (ROC-AUC ≈ 0.94; F1 ≈ 0.62 at threshold 0.50).

## Quickstart
1) Clone and create env
```bash
git clone <repo-url>
cd CMSE492FinalProjectMorof
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
2) Train optimized models (logreg, RF, XGB)
```bash
python -m src.model_optimized
```
Outputs: tuned models in `models/`, metrics in `data/processed/model_results_optimized_misprice.json`.

3) Run threshold sweep for optimized XGB
```bash
python -m src.threshold_tuning
```
Outputs: `data/processed/threshold_sweep_xgb_misprice.csv`, printed best thresholds (F1 ≈ 0.50; precision-oriented ≈ 0.70).

4) Launch diagnostics & backtest notebook
```bash
jupyter lab notebooks/model_diagnostics_and_backtest.ipynb
# or: jupyter notebook notebooks/model_diagnostics_and_backtest.ipynb
```
Notebook regenerates ROC/PR, confusion matrices (t=0.50, 0.70), calibration, SHAP, feature importances, cost-sensitive curve, and PnL plots (saved to `figures/`).

## Core scripts
- `src/features.py`: fetch resolved markets, price history, trades; build raw feature snapshots; writes `data/processed/training_data.csv`.
- `src.clean_data.py`: cleanse/impute; writes `data/processed/training_data_clean.csv`.
- `src.model_simple.py`: baseline models; writes `data/processed/model_results_simple_misprice.json` and baseline `.joblib` files.
- `src.model_optimized.py`: randomized hyperparameter search for logreg/RF/XGB; writes tuned `.joblib` models and `data/processed/model_results_optimized_misprice.json`.
- `src.threshold_tuning.py`: sweep thresholds on optimized XGB; writes `data/processed/threshold_sweep_xgb_misprice.csv`.
- `src.explore_data.py`: quick EDA and correlation heatmap.

## Data and models
- Clean training data: `data/processed/training_data_clean.csv` (7,558 rows, 34 columns; mispricing positives ≈3.7%).
- Saved models: `models/*_misprice.joblib` (baseline) and `*_optimized_misprice.joblib` (tuned).
- Threshold sweep: `data/processed/threshold_sweep_xgb_misprice.csv`.

## Notebook
- `notebooks/model_diagnostics_and_backtest.ipynb`: rebuilds `y_misprice`, loads `models/xgboost_optimized_misprice.joblib`, evaluates held-out split, plots diagnostics, SHAP, and runs optimistic PnL backtest for thresholds 0.50 and 0.70. Paths are relative to the repo root (`../data`, `../models`, `../figures`).

## Key results (optimized XGB)
- ROC-AUC: 0.9428
- F1 @ 0.50: 0.6195 (precision 0.6140, recall 0.6250)
- Precision-oriented @ 0.70: precision 0.6383, recall 0.5357, F1 0.5825
- Backtest: positive expected PnL for signals at 0.50 and 0.70 (see figures).

## Minimal structure
- `src/` — data fetch, feature build, cleaning, modeling, threshold sweep
- `data/processed/` — cleaned dataset, model metrics, threshold sweep
- `models/` — saved baseline and optimized models
- `notebooks/` — diagnostics and backtesting
- `figures/` — all generated plots
- `report/` — LaTeX report (`final_report.tex`), bibliography (`references.bib`), compiled `final_report.pdf`
- `requirements.txt` — Python deps (pandas, numpy, requests, scikit-learn, xgboost, shap, seaborn, matplotlib, jupyter)

## Citation and disclaimer
Course project for CMSE 492, Michigan State University. Data from public Polymarket APIs. For academic use only; not financial advice.
