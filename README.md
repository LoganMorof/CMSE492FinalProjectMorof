# CMSE 492 Polymarket ML Capstone

Using Polymarket public APIs (Gamma + CLOB price history) to collect historical YES/NO market data, engineer features, train models to estimate fair outcome probabilities, and flag mispriced active markets. Models: Logistic Regression (L2), Random Forest/Gradient Boosting, MLP; metrics: Brier Score, Log-Loss, Accuracy; interpretation with SHAP.

## Setup
1) Create venv  
   - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`  
   - Windows (PowerShell): `python -m venv .venv && .venv\Scripts\Activate.ps1`
2) Upgrade pip: `python -m pip install --upgrade pip`
3) Install deps: `python -m pip install -r requirements.txt`
4) (Optional) Launch notebooks: `jupyter notebook` or `jupyter lab`

## Workflow (high level)
- Data fetching: `src/data_fetch.py` to pull from Polymarket APIs; store raw in `data/raw/`.
- Feature engineering: `src/features.py` for price/volume/time-to-resolution features (≥5).
- Modeling: `src/models.py` for logistic regression, tree/boosting, and MLP pipelines; evaluate with Brier, log-loss, accuracy; apply SHAP to best model.
- Outputs: save processed sets to `data/processed/`; notebooks live in `notebooks/`.
