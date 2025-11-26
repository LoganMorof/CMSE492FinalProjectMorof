from __future__ import annotations

import time
from typing import Dict, List, Optional

import pandas as pd
import requests

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"


def fetch_markets_page(offset: int = 0, limit: int = 100, closed: bool = True, timeout: int = 30) -> List[Dict]:
    """
    Fetch a single page of markets from the Gamma API.

    Args:
        offset: Offset for pagination (multiples of limit).
        limit: Number of markets to fetch.
        closed: Whether to fetch closed/resolved markets.
        timeout: HTTP request timeout in seconds.

    Returns:
        A list of market dictionaries. Returns [] on HTTP/JSON errors.
    """
    params = {"offset": offset, "limit": limit, "closed": str(closed).lower()}
    url = f"{GAMMA_BASE_URL}/markets"
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        print("Warning: Unexpected markets payload shape; expected list.")
        return []
    except Exception as exc:
        print(f"Warning: failed to fetch markets page (offset={offset}, limit={limit}): {exc}")
        return []


def fetch_resolved_markets(max_markets: int = 300, page_size: int = 100, sleep_s: float = 0.2) -> List[Dict]:
    """
    Fetch resolved/closed markets in pages until reaching max_markets or no more data.

    Args:
        max_markets: Maximum number of markets to collect.
        page_size: Page size per request.
        sleep_s: Seconds to sleep between page requests.

    Returns:
        List of market dictionaries (may be shorter than max_markets if API exhausts).
    """
    markets: List[Dict] = []
    offset = 0
    while len(markets) < max_markets:
        page = fetch_markets_page(offset=offset, limit=page_size, closed=True)
        if not page:
            break
        markets.extend(page)
        offset += page_size
        if sleep_s > 0:
            time.sleep(sleep_s)
    return markets[:max_markets]


def fetch_price_history(token_id: str, interval: str = "all", fidelity: int = 60, timeout: int = 30) -> List[Dict]:
    """
    Fetch price history for a given token from the CLOB API.

    Args:
        token_id: Token ID from market clobTokenIds.
        interval: Interval range (e.g., 'all').
        fidelity: Granularity in minutes (e.g., 60 for 1h buckets).
        timeout: HTTP request timeout in seconds.

    Returns:
        List of history points (dicts). Returns [] on HTTP/JSON errors.
    """
    params = {"interval": interval, "market": token_id, "fidelity": fidelity}
    url = f"{CLOB_BASE_URL}/prices-history"
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        history = data.get("history", [])
        if isinstance(history, list):
            return history
        print("Warning: Unexpected history payload shape; expected list.")
        return []
    except Exception as exc:
        print(f"Warning: failed to fetch price history for token {token_id}: {exc}")
        return []


def markets_to_dataframe(markets: List[Dict]) -> pd.DataFrame:
    """
    Normalize a list of market dictionaries into a DataFrame.

    Args:
        markets: List of market dicts.

    Returns:
        pandas DataFrame (empty if input is empty).
    """
    if not markets:
        return pd.DataFrame()
    return pd.json_normalize(markets)


def price_history_to_dataframe(history: List[Dict]) -> pd.DataFrame:
    """
    Convert a price history list into a DataFrame with timestamp/price columns.

    Args:
        history: List of history points, typically with keys like 't' (timestamp) and 'p' (price).

    Returns:
        pandas DataFrame with columns ['timestamp', 'price'] (empty if input is empty).
    """
    if not history:
        return pd.DataFrame(columns=["timestamp", "price"])
    df = pd.DataFrame(history)
    rename_map = {}
    if "t" in df.columns:
        rename_map["t"] = "timestamp"
    if "p" in df.columns:
        rename_map["p"] = "price"
    if rename_map:
        df = df.rename(columns=rename_map)
    cols = [c for c in ["timestamp", "price"] if c in df.columns]
    return df[cols] if cols else df


if __name__ == "__main__":
    # Simple smoke test; not for production use.
    print("Fetching resolved markets (up to 50)...")
    markets = fetch_resolved_markets(max_markets=50, page_size=50, sleep_s=0.2)
    mdf = markets_to_dataframe(markets)
    print(f"Markets fetched: {len(markets)}; DataFrame shape: {mdf.shape}")
    if not mdf.empty:
        print(mdf.head())

    # Fetch price history for the first market with a valid token (non-empty, no brackets/quotes).
    token_id: Optional[str] = None
    for m in markets:
        tokens = m.get("clobTokenIds") or []
        for tok in tokens:
            if not isinstance(tok, str):
                continue
            candidate = tok.strip().strip('"').strip("'")
            if candidate and candidate not in {"[", "]", '"', "'"}:
                token_id = candidate
                break
        if token_id:
            break

    if token_id:
        print(f"\nFetching price history for token: {token_id}")
        history = fetch_price_history(token_id=token_id, interval="all", fidelity=60)
        hdf = price_history_to_dataframe(history)
        print(f"Price points returned: {len(history)}; DataFrame shape: {hdf.shape}")
        if not hdf.empty:
            print(hdf.head())
    else:
        print("\nNo valid token IDs found in fetched markets; skipping price history fetch.")
