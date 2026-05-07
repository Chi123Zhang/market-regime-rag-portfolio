"""
data_loader.py
==============
Download price (yfinance) and macro (FRED) data, then persist to parquet so
downstream stages don't re-hit the network.

Public API
----------
    load_prices()    -> pd.DataFrame   # adj-close columns, daily, business days
    load_macro()     -> pd.DataFrame   # FRED series, daily, forward-filled
    load_all()       -> pd.DataFrame   # joined prices + macro on trading days

All loaders are idempotent and use parquet caching:
    data_cache/prices.parquet
    data_cache/macro.parquet

Re-running a loader after the cache file exists returns the cached frame
unless `force_refresh=True`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import (
    START_DATE,
    END_DATE,
    TICKERS,
    FRED_SERIES,
    CACHE_DIR,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Prices (Yahoo Finance)
# --------------------------------------------------------------------------
def _download_prices() -> pd.DataFrame:
    """Pull adjusted-close prices for every ticker in TICKERS."""
    import yfinance as yf  # imported lazily so unit tests can monkeypatch

    log.info("Downloading %d tickers from Yahoo Finance ...", len(TICKERS))
    raw = yf.download(
        tickers=list(TICKERS.keys()),
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,    # we want the explicit Adj Close column
        progress=False,
        group_by="ticker",
    )

    # yfinance returns a multi-index column frame when multiple tickers are
    # requested. Flatten to a single DataFrame of adjusted closes.
    frames = {}
    for tkr in TICKERS:
        if tkr in raw.columns.get_level_values(0):
            frames[tkr] = raw[tkr]["Adj Close"]
        else:
            log.warning("Ticker %s missing from Yahoo response", tkr)
    out = pd.DataFrame(frames).sort_index()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out


def load_prices(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns a daily DataFrame of adjusted-close prices, columns = TICKERS keys.

    Cached at data_cache/prices.parquet.
    """
    cache_path = CACHE_DIR / "prices.parquet"
    if cache_path.exists() and not force_refresh:
        log.info("Loading cached prices from %s", cache_path)
        return pd.read_parquet(cache_path)

    df = _download_prices()
    df.to_parquet(cache_path)
    log.info("Saved %d rows of prices -> %s", len(df), cache_path)
    return df


# --------------------------------------------------------------------------
# Macro (FRED)
# --------------------------------------------------------------------------
def _download_macro() -> pd.DataFrame:
    """Pull every FRED series; forward-fill monthly series to daily."""
    from pandas_datareader import data as pdr  # lazy import

    log.info("Downloading %d series from FRED ...", len(FRED_SERIES))
    out = pdr.DataReader(
        list(FRED_SERIES.keys()),
        "fred",
        start=START_DATE,
        end=END_DATE,
    )
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"

    # Reindex onto a business-day grid and forward-fill monthly series so they
    # can be joined to daily prices. We ffill (not bfill) — at trading day t we
    # know the most recently *released* macro number, never a future one.
    bday_idx = pd.bdate_range(START_DATE, END_DATE)
    out = out.reindex(bday_idx).ffill()
    out.index.name = "date"

    # Derived: 1y CPI YoY % change (will be useful as a feature)
    if "CPIAUCSL" in out.columns:
        out["CPI_YOY"] = out["CPIAUCSL"].pct_change(252) * 100

    return out


def load_macro(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns a daily DataFrame of FRED series, forward-filled.

    Cached at data_cache/macro.parquet.
    """
    cache_path = CACHE_DIR / "macro.parquet"
    if cache_path.exists() and not force_refresh:
        log.info("Loading cached macro from %s", cache_path)
        return pd.read_parquet(cache_path)

    df = _download_macro()
    df.to_parquet(cache_path)
    log.info("Saved %d rows of macro -> %s", len(df), cache_path)
    return df


# --------------------------------------------------------------------------
# Joined frame
# --------------------------------------------------------------------------
def load_all(force_refresh: bool = False) -> pd.DataFrame:
    """
    Inner-join prices and macro on trading days.

    Returns a single DataFrame indexed by date with both price and macro
    columns. This is the canonical input to market_windows.py.
    """
    prices = load_prices(force_refresh=force_refresh)
    macro  = load_macro(force_refresh=force_refresh)
    df = prices.join(macro, how="inner")
    df = df.dropna(how="all")
    return df


# --------------------------------------------------------------------------
# CLI entry point so this can be run standalone:  python data_loader.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = load_all(force_refresh=False)
    print(df.tail())
    print("Shape:", df.shape)
    print("Date range:", df.index.min(), "→", df.index.max())
