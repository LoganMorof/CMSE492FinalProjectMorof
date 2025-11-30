from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "data/processed/training_data_clean.csv"
RESULTS_PATH = "data/processed/model_results_optimized_misprice.json"
MODELS_DIR = "models"


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load cleaned dataset, build mispricing target, and return X, y."""
    if not os.path.exists(INPUT_PATH):
        print(f"Error: input file not found at {INPUT_PATH}")
        raise SystemExit(1)
    df = pd.read_csv(INPUT_PATH)

    required_cols = {"last_price", "final_price"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: required columns missing: {missing}")
        raise SystemExit(1)

    gap = df["final_price"] - df["last_price"]
    y = (gap.abs() >= 0.15).astype(int)
    df["y_misprice"] = y

    X = df.drop(columns=["y", "final_price", "y_misprice"], errors="ignore")

    print("Mispricing label distribution (y_misprice):")
    print(y.value_counts(normalize=False))
    print(y.value_counts(normalize=True))
    return X, y


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Compute evaluation metrics for a fitted model."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_scores = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_scores)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }
    print(f"\n{name} metrics (mispricing):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return metrics


def tune_log_reg(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    param_distributions = {
        "clf__C": loguniform(1e-3, 1e3),
        "clf__penalty": ["l2"],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    param_distributions = {
        "n_estimators": [200, 500, 800],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 0.5],
    }
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError:
        print("xgboost not installed; skipping XGBClassifier.")
        return None

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    base_spw = float(neg) / float(pos) if pos > 0 else 1.0

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    param_distributions = {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "scale_pos_weight": [base_spw * 0.5, base_spw, base_spw * 2],
    }
    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def main() -> None:
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results: Dict[str, Dict[str, Any]] = {}
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Logistic Regression
    print("\nTuning Logistic Regression...")
    log_reg_search = tune_log_reg(X_train, y_train)
    log_reg_best = log_reg_search.best_estimator_
    results["logistic_regression"] = {
        "metrics": evaluate_model("Logistic Regression", log_reg_best, X_test, y_test),
        "best_params": log_reg_search.best_params_,
    }
    joblib.dump(log_reg_best, os.path.join(MODELS_DIR, "logistic_regression_optimized_misprice.joblib"))

    # Random Forest
    print("\nTuning Random Forest...")
    rf_search = tune_random_forest(X_train, y_train)
    rf_best = rf_search.best_estimator_
    results["random_forest"] = {
        "metrics": evaluate_model("Random Forest", rf_best, X_test, y_test),
        "best_params": rf_search.best_params_,
    }
    joblib.dump(rf_best, os.path.join(MODELS_DIR, "random_forest_optimized_misprice.joblib"))

    # XGBoost (optional)
    print("\nTuning XGBoost...")
    xgb_search = tune_xgboost(X_train, y_train)
    if xgb_search is not None:
        xgb_best = xgb_search.best_estimator_
        results["xgboost"] = {
            "metrics": evaluate_model("XGBoost", xgb_best, X_test, y_test),
            "best_params": xgb_search.best_params_,
        }
        joblib.dump(xgb_best, os.path.join(MODELS_DIR, "xgboost_optimized_misprice.joblib"))
    else:
        results["xgboost"] = {"skipped": True}

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {RESULTS_PATH}")

    print("\nSummary (best f1 by model):")
    for name, info in results.items():
        if "metrics" in info:
            print(f"  {name}: f1 = {info['metrics'].get('f1', float('nan')):.4f}")
        else:
            print(f"  {name}: skipped")


if __name__ == "__main__":
    main()
