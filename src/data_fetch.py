from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
HISTORY_CACHE_DIR = "data/raw/history"


def _extract_token_ids(tokens_field) -> List[str]:
    """
    Normalize clobTokenIds into a list of clean token id strings.
    Handles lists, nested lists, and stringified lists like '["id1","id2"]' or 'id1, id2'.
    """
    tokens: List[str] = []

    def clean(s: str) -> str:
        return s.strip().strip('[]"\' ').replace(" ", "")

    if isinstance(tokens_field, str):
        stripped = tokens_field.strip().strip("[]")
        parts = re.split(r"[,\s]+", stripped)
        tokens.extend([clean(p) for p in parts if clean(p)])
    elif isinstance(tokens_field, (list, tuple)):
        for item in tokens_field:
            if isinstance(item, str):
                tokens.append(clean(item))
            elif isinstance(item, (list, tuple)) and item:
                inner = item[0]
                if isinstance(inner, str):
                    tokens.append(clean(inner))

    tokens = [t for t in tokens if t and t not in {"[", "]", '"', "'"}]
    return tokens


def _extract_token_id(tokens_field) -> Optional[str]:
    ids = _extract_token_ids(tokens_field)
    return ids[0] if ids else None


def fetch_markets_page(
    offset: int = 0,
    limit: int = 100,
    closed: bool = True,
    timeout: int = 30,
    extra_params: Optional[Dict] = None,
) -> List[Dict]:
    """Fetch a single page of markets from Gamma."""
    params = {"offset": offset, "limit": limit, "closed": str(closed).lower()}
    if extra_params:
        params.update(extra_params)
    url = f"{GAMMA_BASE_URL}/markets"
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        print(f"[debug] markets page status={resp.status_code} params={params}")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        print("Warning: unexpected markets payload (not a list).")
        return []
    except Exception as exc:
        print(f"Warning: failed to fetch markets page offset={offset}, limit={limit}: {exc}")
        return []


def fetch_resolved_markets(
    max_markets: int = 500,
    page_size: int = 100,
    sleep_s: float = 0.2,
    start_offset: int = 0,
    timeout: int = 30,
    extra_params: Optional[Dict] = None,
) -> List[Dict]:
    """
    Page through resolved markets from Gamma until max_markets or empty page.
    start_offset lets you skip older markets to reach newer CLOB-backed ones.
    Defaults to ordering by volume24hrClob desc and enableOrderBook=true to prefer liquid CLOB markets.
    """
    if extra_params is None:
        extra_params = {
            "enableOrderBook": "true",
            "order": "volume24hrClob",
            "ascending": "false",
        }
    markets: List[Dict] = []
    offset = start_offset
    while len(markets) < max_markets:
        page = fetch_markets_page(
            offset=offset, limit=page_size, closed=True, timeout=timeout, extra_params=extra_params
        )
        if not page:
            break
        markets.extend(page)
        offset += page_size
        if sleep_s > 0:
            time.sleep(sleep_s)
    return markets[:max_markets]


def _cache_path(token_id: str) -> str:
    return os.path.join(HISTORY_CACHE_DIR, f"{token_id}.json")


def fetch_price_history(
    token_id: str,
    interval: str = "all",
    fidelity: int = 60,
    timeout: int = 10,
    sleep_s: float = 0.2,
    max_retries: int = 3,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
) -> List[Dict]:
    """
    Fetch price history for a token from CLOB with caching and gentle retries.
    Defaults to interval='all'. If interval is None, uses a fallback window sequence.
    """
    os.makedirs(HISTORY_CACHE_DIR, exist_ok=True)
    cache_file = _cache_path(token_id)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and "history" in cached and isinstance(cached["history"], list):
                return cached["history"]
            if isinstance(cached, list):
                return cached
        except Exception as exc:
            print(f"Warning: failed to read cache for {token_id}: {exc}")

    now = int(time.time())
    if end_ts is None:
        end_ts = now
    url = f"{CLOB_BASE_URL}/prices-history"
    backoff = sleep_s if sleep_s > 0 else 0.1

    # If interval is provided (default 'all'), try that first
    if interval:
        params = {"market": token_id, "interval": interval, "fidelity": fidelity}
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=timeout)
                print(f"[debug] price-history {token_id} interval={interval} -> {resp.status_code}")
                if resp.status_code == 429:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
                history = data.get("history", [])
                if not isinstance(history, list):
                    history = []
                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                except Exception as exc:
                    print(f"Warning: failed to write cache for {token_id}: {exc}")
                if sleep_s > 0:
                    time.sleep(sleep_s)
                return history
            except Exception as exc:
                print(f"Warning: price history (interval) failed for {token_id} (attempt {attempt}): {exc}")
                time.sleep(backoff)
                backoff *= 2

    # Fallback: try shrinking start/end windows if interval was None or failed
    window_days_sequence = [30, 7, 1]
    for days in window_days_sequence:
        win_start = start_ts if start_ts is not None else end_ts - days * 86400
        params = {"market": token_id, "fidelity": fidelity, "startTs": win_start, "endTs": end_ts}
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=timeout)
                print(f"[debug] price-history {token_id} days={days} -> {resp.status_code}")
                if resp.status_code == 429:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                if resp.status_code == 400 and "interval is too long" in resp.text:
                    break  # shrink window
                resp.raise_for_status()
                data = resp.json()
                history = data.get("history", [])
                if not isinstance(history, list):
                    history = []
                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                except Exception as exc:
                    print(f"Warning: failed to write cache for {token_id}: {exc}")
                if sleep_s > 0:
                    time.sleep(sleep_s)
                return history
            except Exception as exc:
                print(f"Warning: failed (days={days}) for {token_id} (attempt {attempt}): {exc}")
                time.sleep(backoff)
                backoff *= 2
        # try next shorter window
    return []


def markets_to_dataframe(markets: List[Dict]) -> pd.DataFrame:
    if not markets:
        return pd.DataFrame()
    return pd.json_normalize(markets)


def price_history_to_dataframe(history: List[Dict]) -> pd.DataFrame:
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


def find_first_history(
    markets: List[Dict],
    sleep_s: float = 0.2,
    interval: str = "all",
    fidelity: int = 60,
    max_tokens: int = 100,
) -> Tuple[Optional[str], List[Dict]]:
    checked = 0
    for m in markets:
        for tok in _extract_token_ids(m.get("clobTokenIds")):
            if checked >= max_tokens:
                return None, []
            checked += 1
            hist = fetch_price_history(tok, interval=interval, fidelity=fidelity, sleep_s=sleep_s)
            if hist:
                return tok, hist
            else:
                print(f"No history for token {tok}; trying next...")
    return None, []


if __name__ == "__main__":
    offsets_to_try = [0, 5000, 10000, 15000]
    found = False
    for start in offsets_to_try:
        print(f"\nFetching resolved markets (start_offset={start})...")
        mkts = fetch_resolved_markets(
            max_markets=200, page_size=100, sleep_s=0.2, start_offset=start
        )
        print(f"Fetched {len(mkts)} markets.")
        if mkts:
            tok, hist = find_first_history(
                mkts, sleep_s=0.2, interval="all", fidelity=60, max_tokens=100
            )
            if tok and hist:
                hdf = price_history_to_dataframe(hist)
                print(f"\nFound history for token: {tok}")
                print(f"History points: {len(hist)}; DataFrame shape: {hdf.shape}")
                if not hdf.empty:
                    print(hdf.head())
                found = True
                break
            else:
                print("No history found in this batch; advancing offset...")
        else:
            print("No markets returned; stopping.")
    if not found:
        print("\nNo token with history found in sampled offsets; consider increasing start_offset further.")
