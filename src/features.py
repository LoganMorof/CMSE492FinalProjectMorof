from __future__ import annotations

import ast
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data_fetch import (
    fetch_price_history,
    fetch_resolved_markets,
    markets_to_dataframe,
    price_history_to_dataframe,
)


def _parse_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    """Safely parse a timestamp string to pandas.Timestamp (UTC)."""
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return ts if pd.notnull(ts) else None


def _extract_token_ids(tokens_field) -> List[str]:
    """
    Normalize clobTokenIds into a list of clean token id strings.
    Handles stringified lists and nested lists.
    """
    tokens: List[str] = []

    def clean(s: str) -> str:
        return s.strip().strip('[]"\' ').replace(" ", "")

    if isinstance(tokens_field, str):
        stripped = tokens_field.strip().strip("[]")
        parts = stripped.split(",")
        tokens.extend([clean(p) for p in parts if clean(p)])
    elif isinstance(tokens_field, (list, tuple)):
        for item in tokens_field:
            if isinstance(item, str):
                tokens.append(clean(item))
            elif isinstance(item, (list, tuple)) and item:
                inner = item[0]
                if isinstance(inner, str):
                    tokens.append(clean(inner))
    tokens = [t for t in tokens if t]
    return tokens


def _fallback_price_from_market(market: Dict) -> Optional[float]:
    """
    Try to derive a last price from market metadata if history is empty.
    Uses outcomePrices or lastTradePrice if available.
    """
    prices = market.get("outcomePrices") or market.get("outcome_prices")
    if isinstance(prices, (list, tuple)) and prices:
        try:
            return float(prices[0])
        except Exception:
            return None
    ltp = market.get("lastTradePrice")
    try:
        return float(ltp) if ltp is not None else None
    except Exception:
        return None


def _extract_label(market: Dict) -> Optional[int]:
    """
    Extract a binary label. Tries common outcome fields; falls back to lastTradePrice heuristic if market is closed.
    """
    keys = [
        "winningOutcome",
        "resolvedOutcome",
        "winning_outcome",
        "resolved_outcome",
        "winningSide",
        "winning_side",
        "result",
        "outcome",
        "resolution",
    ]
    for k in keys:
        if k in market:
            val = market.get(k)
            if isinstance(val, bool):
                return 1 if val else 0
            if isinstance(val, (int, float)) and val in (0, 1):
                return int(val)
            if isinstance(val, str):
                outcome_clean = val.strip().lower()
                if outcome_clean in {"yes", "y", "1", "true"}:
                    return 1
                if outcome_clean in {"no", "n", "0", "false"}:
                    return 0

    # Fallback: if market is closed, use lastTradePrice heuristic as a proxy label.
    if market.get("closed") is True:
        ltp = _fallback_price_from_market(market)
        if ltp is not None:
            return 1 if ltp >= 0.5 else 0
    # TODO: replace heuristic with actual resolved outcome when available in the payload.
    return None


def compute_market_features(
    market: Dict,
    history_df: pd.DataFrame,
    snapshot_offset_days: int = 3,
    min_history_points: int = 1,
) -> Optional[Dict]:
    """
    Compute snapshot features for a single market using its price history and metadata.
    Returns a feature dict including target 'y', or None if not enough data/label.
    """
    df = history_df.copy()
    if "timestamp" not in df.columns and "t" in df.columns:
        df = df.rename(columns={"t": "timestamp"})
    if "price" not in df.columns and "p" in df.columns:
        df = df.rename(columns={"p": "price"})

    df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    resolution_time = (
        _parse_timestamp(market.get("endDateIso"))
        or _parse_timestamp(market.get("closeTime"))
        or _parse_timestamp(market.get("endTime"))
    )
    start_time = _parse_timestamp(market.get("startDateIso")) or _parse_timestamp(market.get("startTime"))
    if resolution_time is None or start_time is None:
        return None

    snapshot_time = resolution_time - pd.Timedelta(days=snapshot_offset_days)
    df = df[df["timestamp"] <= snapshot_time].sort_values("timestamp")

    if "price" not in df.columns or df.empty or len(df) < min_history_points:
        last_price = _fallback_price_from_market(market)
        ma_7d = np.nan
        vol_7d = np.nan
        vol_30d = np.nan
        price_trend_7d = np.nan
    else:
        def window_stats(days: int) -> pd.DataFrame:
            window_start = snapshot_time - pd.Timedelta(days=days)
            return df[df["timestamp"] >= window_start]

        last_price = df["price"].iloc[-1]
        w7 = window_stats(7)
        w30 = window_stats(30)
        ma_7d = w7["price"].mean() if not w7.empty else np.nan
        vol_7d = w7["price"].std(ddof=0) if not w7.empty else np.nan
        vol_30d = w30["price"].std(ddof=0) if not w30.empty else np.nan
        price_7d_ago = w7["price"].iloc[0] if len(w7) > 0 else np.nan
        price_trend_7d = last_price - price_7d_ago if pd.notnull(price_7d_ago) else np.nan

    age_days = (snapshot_time - start_time).total_seconds() / 86400.0
    time_to_resolution_days = (resolution_time - snapshot_time).total_seconds() / 86400.0

    y = _extract_label(market)
    if y is None:
        return None

    return {
        "market_id": market.get("id"),
        "last_price": last_price,
        "ma_7d": ma_7d,
        "price_trend_7d": price_trend_7d,
        "vol_7d": vol_7d,
        "vol_30d": vol_30d,
        "age_days": age_days,
        "time_to_resolution_days": time_to_resolution_days,
        "category": market.get("category"),
        "y": y,
    }


def build_training_dataset(
    markets: List[Dict],
    snapshot_offset_days_list: Sequence[int] = (7, 3, 1),
    max_markets: Optional[int] = None,
    min_history_points: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build a training DataFrame by fetching histories, computing features, and assembling rows.
    Each market can contribute multiple rows (one per snapshot_offset_days entry).
    """
    rows: List[Dict] = []
    to_iterate = markets[:max_markets] if max_markets else markets

    skipped_no_token = 0
    skipped_label = 0

    for market in to_iterate:
        token_ids = _extract_token_ids(market.get("clobTokenIds"))
        if not token_ids:
            skipped_no_token += 1
            continue

        added_for_market = False
        for tok in token_ids:
            history = fetch_price_history(tok, interval="all", fidelity=60)
            hdf = price_history_to_dataframe(history)
            for offset in snapshot_offset_days_list:
                feats = compute_market_features(
                    market=market,
                    history_df=hdf,
                    snapshot_offset_days=offset,
                    min_history_points=min_history_points,
                )
                if feats is not None:
                    feats["snapshot_offset_days"] = offset
                    rows.append(feats)
                    added_for_market = True
            if added_for_market:
                break  # no need to try other tokens once we have rows
        if not added_for_market:
            skipped_label += 1

    if verbose:
        print(
            f"Processed markets: {len(to_iterate)}, kept rows: {len(rows)}, "
            f"no_token: {skipped_no_token}, no_label_or_data: {skipped_label}"
        )

    return pd.DataFrame(rows)


def save_training_dataset_to_csv(
    df: pd.DataFrame,
    path: str = "data/processed/training_data.csv",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    print("Fetching resolved markets...")
    markets = fetch_resolved_markets(
        max_markets=1500,
        page_size=100,
        sleep_s=0.2,
    )
    print(f"Fetched {len(markets)} markets.")
    _mdf = markets_to_dataframe(markets)
    if not _mdf.empty:
        print("Sample markets:")
        print(_mdf.head())

    print("\nBuilding training dataset...")
    df = build_training_dataset(
        markets=markets,
        snapshot_offset_days_list=(7, 3, 1),
        max_markets=None,
        min_history_points=1,
        verbose=True,
    )
    print(f"Training DataFrame shape: {df.shape}")
    if not df.empty:
        print(df.head())
        if "y" in df.columns:
            print("Label distribution:")
            print(df["y"].value_counts())

    output_path = "data/processed/training_data.csv"
    save_training_dataset_to_csv(df, path=output_path)
    print(f"\nSaved training data to: {output_path}")
