from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "data/processed/training_data_clean.csv"
RESULTS_PATH = "data/processed/model_results_simple_misprice.json"
MODELS_DIR = "models"


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load cleaned dataset, build mispricing target, and split into features/target."""
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

    # Drop target and leakage columns from features
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


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """Train baseline models and return fitted instances."""
    models: Dict[str, Any] = {}

    log_reg = Pipeline(
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
    log_reg.fit(X_train, y_train)
    models["logistic_regression"] = log_reg

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    try:
        from xgboost import XGBClassifier  # type: ignore

        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
        )
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb
    except ImportError:
        print("xgboost not installed; skipping XGBClassifier.")

    return models


def main() -> None:
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    baseline_acc = (y_test == 0).mean()
    print(f"Baseline accuracy (always predict no mispricing = 0): {baseline_acc:.4f}")

    models = train_models(X_train, y_train)

    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        results[name] = evaluate_model(name, model, X_test, y_test)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {RESULTS_PATH}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in models.items():
        model_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}_misprice.joblib")
        joblib.dump(model, model_path)
        print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
