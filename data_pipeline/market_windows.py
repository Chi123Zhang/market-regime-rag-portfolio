"""
market_windows.py
=================
Turn a daily price+macro panel into a sequence of *market-window* records.

What is a market window?
------------------------
A market window is a snapshot of market state ending at trading day `t`. It
summarises the previous WINDOW_LONG (~63) trading days through a small set of
numeric features. We emit one window every WINDOW_STEP (~5) trading days.

These windows are the documents that the downstream RAG pipeline indexes:

    [ window @ 2008-09-26 ]  →  embed → FAISS
    [ window @ 2008-10-03 ]  →  embed → FAISS
    [ window @ 2008-10-10 ]  →  embed → FAISS
    ...

Why windows and not raw daily rows?
-----------------------------------
1. Regimes are *persistent* phenomena — a single day's return is too noisy.
2. The LLM can reason over rich textual descriptions of a window much more
   effectively than over a single number.
3. Discretising into windows keeps the corpus size bounded (~1000 docs over
   20 years, vs. 5000+ daily rows).

Look-ahead bias guarantee
-------------------------
Every feature for window ending at `t` uses ONLY data with index <= t.
Macro series are forward-filled, never back-filled. This is verified in
tests/test_no_lookahead.py.

Public API
----------
    build_windows(panel: pd.DataFrame) -> pd.DataFrame
        rows = windows, columns = features. The frame's index is window-end
        date so it can be joined to forward returns for evaluation.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from config import (
    WINDOW_LONG,
    WINDOW_SHORT,
    WINDOW_STEP,
    TRADING_DAYS_PER_YEAR,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Helpers — every helper takes a Series/DataFrame slice covering *exactly*
# the look-back window and returns scalar features. No look-ahead by design.
# --------------------------------------------------------------------------
def _annualised_return(prices: pd.Series) -> float:
    """Compounded annualised return over the window."""
    if len(prices) < 2 or prices.iloc[0] <= 0:
        return np.nan
    total = prices.iloc[-1] / prices.iloc[0] - 1
    years = len(prices) / TRADING_DAYS_PER_YEAR
    return (1 + total) ** (1 / years) - 1 if years > 0 else np.nan


def _annualised_vol(returns: pd.Series) -> float:
    """Annualised stdev of daily simple returns."""
    if returns.dropna().size < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(prices: pd.Series) -> float:
    """Worst peak-to-trough drawdown observed inside the window."""
    if prices.empty:
        return np.nan
    running_max = prices.cummax()
    dd = prices / running_max - 1
    return float(dd.min())


def _sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe with rf=0 (we don't have a clean rf series here)."""
    s = returns.dropna()
    if s.size < 2 or s.std() == 0:
        return np.nan
    return float(s.mean() / s.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _drawdown_from_1y_high(prices_1y: pd.Series) -> float:
    """
    Current price as a % of the rolling 1-year high. Used by the bear-regime
    rule. `prices_1y` should be the trailing 252 trading-day slice.
    """
    if prices_1y.empty:
        return np.nan
    return float(prices_1y.iloc[-1] / prices_1y.max() - 1)


def _correlation(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation of two daily-return series; NaN-safe."""
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 5:
        return np.nan
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


# --------------------------------------------------------------------------
# Window iterator
# --------------------------------------------------------------------------
def _window_end_dates(index: pd.DatetimeIndex) -> Iterable[pd.Timestamp]:
    """
    Yield window-end timestamps. We start once we have enough history for the
    longest look-back AND the 252-day rolling high used by the bear rule.
    """
    min_history = max(WINDOW_LONG, 252)
    for pos in range(min_history - 1, len(index), WINDOW_STEP):
        yield index[pos]


# --------------------------------------------------------------------------
# Feature builder for a single window
# --------------------------------------------------------------------------
def _features_for_window(panel: pd.DataFrame, t: pd.Timestamp) -> dict:
    """
    Compute every feature for the window ending at trading day `t`.

    Uses ONLY rows with index <= t.
    """
    # Slice once, pass slices to helpers — keeps look-ahead impossible.
    history = panel.loc[:t]
    long_   = history.iloc[-WINDOW_LONG:]
    short_  = history.iloc[-WINDOW_SHORT:]
    year_   = history.iloc[-252:] if len(history) >= 252 else history

    spy = long_["SPY"]
    spy_short = short_["SPY"]
    spy_year  = year_["SPY"]
    spy_ret_d = spy.pct_change()
    spy_ret_d_short = spy_short.pct_change()

    # Equity-bond correlation (60d) — proxy for "risk-off" cross-asset behaviour
    agg_ret_d = long_["AGG"].pct_change()
    eq_bd_corr = _correlation(spy_ret_d, agg_ret_d)

    # Credit spread proxy: HYG / LQD price ratio rate of change.
    # Falling HYG vs LQD => credit stress. We measure 21d log-change.
    hyg = short_["HYG"]; lqd = short_["LQD"]
    if hyg.notna().all() and lqd.notna().all() and len(hyg) > 1:
        credit_proxy_chg = float(np.log(hyg.iloc[-1] / lqd.iloc[-1])
                                 - np.log(hyg.iloc[0] / lqd.iloc[0]))
    else:
        credit_proxy_chg = np.nan

    feats = {
        "window_end":   t,
        "window_start": long_.index[0],
        "n_days":       len(long_),

        # Equity (long window)
        "spy_ret_63d":      _annualised_return(spy),
        "spy_vol_63d":      _annualised_vol(spy_ret_d),
        "spy_sharpe_63d":   _sharpe(spy_ret_d),
        "spy_maxdd_63d":    _max_drawdown(spy),

        # Equity (short window — recent state)
        "spy_ret_21d":      _annualised_return(spy_short),
        "spy_vol_21d":      _annualised_vol(spy_ret_d_short),

        # Drawdown vs trailing 1y high (used by bear rule)
        "spy_dd_from_1y_high": _drawdown_from_1y_high(spy_year),

        # 200d MA flag — using the long_ slice plus a longer trailing slice
        "spy_above_200ma": float(
            history["SPY"].iloc[-1] > history["SPY"].iloc[-200:].mean()
        ) if len(history) >= 200 else np.nan,

        # Volatility regime (VIX) — current level + 21d change
        "vix_last":         float(history["^VIX"].iloc[-1])
                            if "^VIX" in history else np.nan,
        "vix_21d_change":   float(history["^VIX"].iloc[-1]
                                  - history["^VIX"].iloc[-WINDOW_SHORT])
                            if "^VIX" in history and len(history) > WINDOW_SHORT
                            else np.nan,

        # Cross-asset
        "eq_bd_corr_63d":       eq_bd_corr,
        "credit_proxy_chg_21d": credit_proxy_chg,

        # Macro snapshot (point-in-time, ffilled)
        "yield_curve_10y2y":  float(history["T10Y2Y"].iloc[-1])
                              if "T10Y2Y" in history else np.nan,
        "ten_year_yield":     float(history["DGS10"].iloc[-1])
                              if "DGS10" in history else np.nan,
        "fed_funds":          float(history["DFF"].iloc[-1])
                              if "DFF" in history else np.nan,
        "credit_spread_oas":  float(history["BAMLH0A0HYM2"].iloc[-1])
                              if "BAMLH0A0HYM2" in history else np.nan,
        "cpi_yoy":            float(history["CPI_YOY"].iloc[-1])
                              if "CPI_YOY" in history else np.nan,
        "unrate":             float(history["UNRATE"].iloc[-1])
                              if "UNRATE" in history else np.nan,
        "nber_recession":     float(history["USREC"].iloc[-1])
                              if "USREC" in history else np.nan,
    }
    return feats


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def build_windows(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full set of market windows from a daily price+macro panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of data_loader.load_all(). Must contain at minimum:
        SPY, AGG, HYG, LQD, ^VIX columns plus the macro columns referenced
        in _features_for_window.

    Returns
    -------
    pd.DataFrame
        One row per window, indexed by `window_end`. Columns are the features
        described in _features_for_window.
    """
    panel = panel.sort_index()
    rows = [_features_for_window(panel, t) for t in _window_end_dates(panel.index)]
    out = pd.DataFrame(rows).set_index("window_end").sort_index()
    log.info("Built %d windows (%s → %s)",
             len(out), out.index.min().date(), out.index.max().date())
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    from data_loader import load_all
    from config import OUTPUT_DIR

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    panel = load_all()
    windows = build_windows(panel)
    out_path = OUTPUT_DIR / "windows.parquet"
    windows.to_parquet(out_path)
    print(f"Saved {len(windows)} windows -> {out_path}")
    print(windows.tail())
