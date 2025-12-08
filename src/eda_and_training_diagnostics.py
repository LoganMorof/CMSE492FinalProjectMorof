from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Paths
DATA_PATH = Path("data/processed/training_data_clean.csv")
PROC_DIR = Path("data/processed")
FIG_DIR = Path("figures")

# Plot style
sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (8, 5)


def _ensure_dirs() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    print("=== Loading data ===")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if {"last_price", "final_price"} - set(df.columns):
        raise ValueError("Required columns last_price/final_price missing")
    gap = df["final_price"] - df["last_price"]
    df["y_misprice"] = (gap.abs() >= 0.15).astype(int)
    print(f"Shape: {df.shape}")
    print(df.dtypes)
    print(df.head())
    return df


def missingness_and_class_balance(df: pd.DataFrame) -> None:
    print("=== Missingness & class balance ===")
    missing = df.isna().sum().to_frame("n_missing")
    missing["frac_missing"] = missing["n_missing"] / len(df)
    missing.to_csv(PROC_DIR / "missingness_summary.csv", index=True)
    print(missing.head())

    counts = df["y_misprice"].value_counts().sort_index()
    props = df["y_misprice"].value_counts(normalize=True).sort_index()
    print("Class counts:\n", counts)
    print("Class proportions:\n", props)

    plt.figure()
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.xlabel("y_misprice")
    plt.ylabel("count")
    plt.title("Class Counts (y_misprice)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_balance_counts_v2.png", dpi=300)

    plt.figure()
    sns.barplot(x=props.index, y=props.values, palette="magma")
    plt.xlabel("y_misprice")
    plt.ylabel("proportion")
    plt.title("Class Proportions (y_misprice)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_balance_proportions_v2.png", dpi=300)


def univariate_distributions(df: pd.DataFrame) -> None:
    print("=== Univariate distributions ===")
    features = [
        "last_price",
        "final_price",
        "recent_vol",
        "recent_ma",
        "age_days",
        "time_to_resolution_days",
        "trades_15m",
        "net_order_flow_15m",
        "snapshot_offset_hours",
    ]
    stats_rows: List[Dict] = []
    for feat in features:
        if feat not in df.columns:
            print(f"Skipping missing feature: {feat}")
            continue
        series = df[feat]
        stats = series.describe()
        stats_rows.append({"feature": feat, **stats.to_dict()})

        plt.figure()
        sns.histplot(series, kde=True, bins=40, color="steelblue")
        plt.title(f"Distribution of {feat}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"univariate_{feat}.png", dpi=300)

        plt.figure()
        sns.kdeplot(data=df, x=feat, hue="y_misprice", common_norm=False, fill=True, alpha=0.4)
        plt.title(f"{feat} by mispricing label")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"univariate_{feat}_by_misprice.png", dpi=300)

    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(PROC_DIR / "univariate_summary_stats.csv", index=False)


def bivariate_plots(df: pd.DataFrame) -> None:
    print("=== Bivariate plots ===")
    df_plot = df.copy()
    if {"last_price", "final_price", "y_misprice"} <= set(df_plot.columns):
        plt.figure()
        sns.scatterplot(
            data=df_plot,
            x="last_price",
            y="final_price",
            hue="y_misprice",
            alpha=0.4,
            palette="coolwarm",
        )
        plt.title("Last Price vs Final Price")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "bivariate_last_vs_final_price.png", dpi=300)

    df_plot["price_gap"] = (df_plot["final_price"] - df_plot["last_price"]).abs()
    if {"time_to_resolution_days", "price_gap", "y_misprice"} <= set(df_plot.columns):
        plt.figure()
        sns.scatterplot(
            data=df_plot,
            x="time_to_resolution_days",
            y="price_gap",
            hue="y_misprice",
            alpha=0.4,
            palette="viridis",
        )
        plt.title("Time to Resolution vs |Final - Last|")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "bivariate_time_to_resolution_vs_gap.png", dpi=300)

    if {"recent_vol", "price_gap", "y_misprice"} <= set(df_plot.columns):
        plt.figure()
        sns.scatterplot(
            data=df_plot,
            x="recent_vol",
            y="price_gap",
            hue="y_misprice",
            alpha=0.4,
            palette="magma",
        )
        plt.title("Recent Volatility vs |Final - Last|")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "bivariate_recent_vol_vs_gap.png", dpi=300)


def correlation_heatmap(df: pd.DataFrame) -> None:
    print("=== Correlation heatmap ===")
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()
    corr.to_csv(PROC_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.7})
    plt.title("Correlation Heatmap (numeric features)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=300)


def train_test_split_stratified(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    print("=== Train/Test split ===")
    y = df["y_misprice"]
    # Drop target, leakage columns, and the plotting-only price_gap if present
    X = df.drop(columns=["y_misprice", "final_price", "y", "price_gap"], errors="ignore")
    if "split" in df.columns:
        test_mask = df["split"] == "test"
        X_train, X_test = X[~test_mask], X[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]
        print(f"Using provided split column. Train: {X_train.shape}, Test: {X_test.shape}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Stratified split. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def get_models(scale_pos_weight: float) -> Dict[str, object]:
    models: Dict[str, object] = {}

    log_reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")),
        ]
    )
    models["logreg"] = log_reg

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    models["random_forest"] = rf

    try:
        from xgboost import XGBClassifier  # type: ignore

        xgb = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        models["xgboost"] = xgb
    except ImportError:
        print("xgboost not installed; skipping XGBClassifier.")
    return models


def run_learning_curve(
    model, X, y, model_name: str, scoring: str = "f1", cv: int = 5, n_jobs: int = -1
) -> None:
    print(f"=== Learning curve: {model_name} ===")
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    lc_df = pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_score": train_mean,
            "val_score": val_mean,
        }
    )
    lc_df.to_csv(PROC_DIR / f"learning_curve_{model_name}.csv", index=False)

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Train F1")
    plt.plot(train_sizes, val_mean, label="Val F1")
    plt.xlabel("Training samples")
    plt.ylabel("F1 score")
    plt.title(f"Learning Curve ({model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"learning_curve_{model_name}.png", dpi=300)


def time_model(model, X_train, y_train, X_test, model_name: str) -> Tuple[float, float]:
    print(f"=== Timing model: {model_name} ===")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t1 = time.perf_counter()
    train_time = t1 - t0

    t0 = time.perf_counter()
    if hasattr(model, "predict_proba"):
        _ = model.predict_proba(X_test)
    else:
        _ = model.predict(X_test)
    t1 = time.perf_counter()
    infer_time_ms_per_1k = (t1 - t0) / len(X_test) * 1000.0

    return train_time, infer_time_ms_per_1k


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict[str, float]:
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    prec = metrics.precision_score(y_test, y_pred, zero_division=0)
    rec = metrics.recall_score(y_test, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)
    acc = metrics.accuracy_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_proba)
    pr_auc = metrics.average_precision_score(y_test, y_proba)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def main() -> None:
    _ensure_dirs()
    df = load_data()
    missingness_and_class_balance(df)
    univariate_distributions(df)
    bivariate_plots(df)
    correlation_heatmap(df)

    X_train, X_test, y_train, y_test = train_test_split_stratified(df)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = float(neg) / float(pos) if pos > 0 else 1.0

    models = get_models(scale_pos_weight=spw)
    runtime_rows: List[Dict[str, float]] = []
    metric_rows: List[Dict[str, float]] = []

    for name, model in models.items():
        run_learning_curve(model, X_train, y_train, name)
        train_time, infer_time = time_model(model, X_train, y_train, X_test, name)
        runtime_rows.append(
            {
                "model_name": name,
                "train_time_seconds": train_time,
                "infer_time_ms_per_1k_samples": infer_time,
                "notes": "optimized hyperparams" if name == "xgboost" else "baseline-ish",
            }
        )
        metrics_row = evaluate_model(model, X_test, y_test, name)
        metric_rows.append(metrics_row)

    if runtime_rows:
        runtime_df = pd.DataFrame(runtime_rows)
        print("\n=== Runtime comparison ===")
        print(runtime_df)
        runtime_df.to_csv(PROC_DIR / "model_runtime_comparison.csv", index=False)

    if metric_rows:
        metrics_df = pd.DataFrame(metric_rows)
        print("\n=== Metrics comparison ===")
        print(metrics_df)
        metrics_df.to_csv(PROC_DIR / "model_metrics_comparison.csv", index=False)


if __name__ == "__main__":
    main()
