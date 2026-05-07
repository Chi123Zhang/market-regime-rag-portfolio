"""
backtest.py
===========
Walk-forward portfolio back-test driven by regime labels.

This is the "Evaluation Report" deliverable from the proposal:

    A thorough back-test report of 2005-2025, plots of regime accuracy versus
    HMM baselines, comparison of Sharpe ratio, analysis of drawdowns ...

and addresses research question RQ3:

    How does the performance of regime-conditioned portfolio strategies
    (in terms of risk-adjusted performance, Sharpe ratio, maximum drawdown)
    compare with the performance of both unconditional allocation strategies
    and momentum-based alternative strategies?

Strategies compared
-------------------
  1.  RAG-Regime    — weights driven by RAG-predicted label at each rebalance
  2.  Oracle-Regime — same weight map, but using rule-based consensus label
                      (upper bound for any regime-conditioned strategy)
  3.  Static 60/40  — unconditional 60% SPY / 40% IEF
  4.  SPY Buy&Hold  — passive equity benchmark
  5.  Momentum      — long SPY when 21-day return > 0 else IEF

Mechanics
---------
  - Universe: SPY (equity), IEF (Treasuries), GLD (gold).
  - Daily prices loaded from data_pipeline/data_cache/prices.parquet.
  - Rebalances on every regime-prediction date.
  - To avoid look-ahead in the back-test itself, weights set at close of date
    `t` are applied to returns from `t+1` onward (the daily regime is shifted
    by 1 day before being multiplied with returns).
  - No transaction cost / slippage in the base run — turnover is reported so
    the reader can apply their own assumption.

Outputs (written under ./outputs/ by default)
---------------------------------------------
  - backtest_daily_returns.csv  daily returns of every strategy
  - backtest_equity_curves.csv  cumulative-return curves (NAV starting at 1.0)
  - backtest_metrics.csv        summary metrics table
  - backtest_drawdowns.csv      drawdown series per strategy
  - equity_curves.png           NAV chart
  - drawdowns.png               drawdown chart
  - regime_timeline.png         predicted-vs-true regime over time
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")           # safe for headless / Colab
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PRICES_PATH = "data_pipeline/data_cache/prices.parquet"

# Regime → portfolio weight map (matches financial_rag.portfolio_rationale).
# Each row sums to 1.0.
REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "bull":     {"SPY": 0.70, "IEF": 0.20, "GLD": 0.10},
    "bear":     {"SPY": 0.30, "IEF": 0.50, "GLD": 0.20},
    "high_vol": {"SPY": 0.40, "IEF": 0.40, "GLD": 0.20},
    "risk_off": {"SPY": 0.20, "IEF": 0.60, "GLD": 0.20},
    "unknown":  {"SPY": 0.50, "IEF": 0.40, "GLD": 0.10},   # neutral fallback
}

ASSETS = ["SPY", "IEF", "GLD"]
TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_returns(prices_path: str = PRICES_PATH) -> pd.DataFrame:
    """Load daily returns for SPY / IEF / GLD (drop pre-GLD-inception rows)."""
    prices = pd.read_parquet(prices_path)[ASSETS].copy()
    prices = prices.dropna()                       # drop pre-GLD rows (~2003-2004)
    rets = prices.pct_change().dropna()
    rets.index = pd.to_datetime(rets.index)
    return rets


# --------------------------------------------------------------------------
# Build daily weight schedules per strategy
# --------------------------------------------------------------------------
def _regime_to_weights(regime: str) -> pd.Series:
    w = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["unknown"])
    return pd.Series(w, index=ASSETS).fillna(0.0)


def regime_weight_schedule(
    regime_series: pd.Series,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Convert a sparse regime series (one entry per rebalance date) into a daily
    weight DataFrame aligned to `daily_index`.

    The weights set at rebalance date `t` are forward-filled from `t` onward,
    then **shifted by one trading day** so that they only affect returns from
    `t+1` — the standard convention to avoid look-ahead in a back-test.
    """
    # Sparse weight frame: one row per rebalance date
    sparse_w = pd.DataFrame(
        [_regime_to_weights(reg) for reg in regime_series.values],
        index=regime_series.index,
        columns=ASSETS,
    )
    # Reindex to daily, forward-fill, then shift by one day
    daily_w = sparse_w.reindex(daily_index).ffill().shift(1)
    return daily_w


def momentum_weight_schedule(
    returns: pd.DataFrame,
    lookback_days: int = 21,
) -> pd.DataFrame:
    """
    Simple momentum baseline:
      - if SPY's trailing-21d return > 0   → 100% SPY
      - else                               → 100% IEF
    """
    spy_mom = returns["SPY"].rolling(lookback_days).sum()
    long_spy = (spy_mom > 0).astype(float).shift(1)  # use yesterday's signal

    w = pd.DataFrame(0.0, index=returns.index, columns=ASSETS)
    w["SPY"] = long_spy
    w["IEF"] = 1.0 - long_spy
    return w


def static_6040_schedule(daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=daily_index, columns=ASSETS)
    w["SPY"] = 0.60
    w["IEF"] = 0.40
    return w


def spy_buy_and_hold(daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=daily_index, columns=ASSETS)
    w["SPY"] = 1.0
    return w


# --------------------------------------------------------------------------
# Portfolio simulation
# --------------------------------------------------------------------------
def simulate(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """Daily portfolio return = sum(w_i * r_i). Aligns and drops missing rows."""
    aligned = weights.reindex(returns.index).fillna(0.0)
    port = (aligned * returns).sum(axis=1)
    # First few rows may be 0 because the weight schedule needs warm-up
    return port


def turnover(weights: pd.DataFrame) -> float:
    """One-sided turnover, summed across days, then annualised."""
    diffs = weights.diff().abs().sum(axis=1).fillna(0.0)
    return float(diffs.sum() / max(len(weights), 1) * TRADING_DAYS)


# --------------------------------------------------------------------------
# Performance metrics
# --------------------------------------------------------------------------
@dataclass
class PerfMetrics:
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    annual_turnover: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "Total Return":     round(self.total_return, 4),
            "CAGR":             round(self.cagr, 4),
            "Ann. Vol":         round(self.ann_vol, 4),
            "Sharpe":           round(self.sharpe, 4),
            "Max Drawdown":     round(self.max_drawdown, 4),
            "Calmar":           round(self.calmar, 4),
            "Hit Rate":         round(self.hit_rate, 4),
            "Ann. Turnover":    round(self.annual_turnover, 4),
        }


def compute_metrics(
    daily_returns: pd.Series,
    weights: Optional[pd.DataFrame] = None,
) -> PerfMetrics:
    r = daily_returns.dropna()
    if len(r) == 0:
        return PerfMetrics(0, 0, 0, 0, 0, 0, 0, 0)

    nav = (1 + r).cumprod()
    total_return = float(nav.iloc[-1] - 1)
    years = len(r) / TRADING_DAYS
    cagr = float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0

    running_peak = nav.cummax()
    drawdown = nav / running_peak - 1
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    hit_rate = float((r > 0).mean())

    ann_to = turnover(weights) if weights is not None else 0.0

    return PerfMetrics(
        total_return=total_return, cagr=cagr, ann_vol=ann_vol, sharpe=sharpe,
        max_drawdown=max_dd, calmar=calmar, hit_rate=hit_rate,
        annual_turnover=ann_to,
    )


def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    nav = (1 + daily_returns.fillna(0)).cumprod()
    return nav / nav.cummax() - 1


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_equity_curves(equity: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in equity.columns:
        ax.plot(equity.index, equity[col], label=col, linewidth=1.5)
    ax.set_title("Equity curves (NAV starting at 1.0)")
    ax.set_ylabel("NAV")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_drawdowns(dds: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in dds.columns:
        ax.plot(dds.index, dds[col], label=col, linewidth=1.2)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_regime_timeline(regime_df: pd.DataFrame, out_path: str) -> None:
    """Predicted vs true regime as colour bands over time."""
    regimes = ["bull", "high_vol", "bear", "risk_off"]
    palette = {"bull": "#2ecc71", "high_vol": "#f1c40f",
               "bear":  "#e74c3c", "risk_off": "#8e44ad",
               "unknown": "#bdc3c7"}

    fig, axes = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
    for ax, col, title in zip(
        axes,
        ["true_label", "pred_label"],
        ["Rule-based consensus (ground truth)", "RAG prediction"],
    ):
        for reg in regimes:
            mask = regime_df[col] == reg
            ax.fill_between(
                regime_df.index, 0, 1,
                where=mask,
                color=palette.get(reg, "#bdc3c7"),
                step="post", label=reg,
            )
        ax.set_yticks([])
        ax.set_title(title)
        ax.set_xlim(regime_df.index.min(), regime_df.index.max())

    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[r]) for r in regimes]
    axes[0].legend(handles, regimes, loc="upper right",
                   ncol=4, fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Top-level runner
# --------------------------------------------------------------------------
def run_backtest(
    regime_labels_path: str = "outputs/regime_labels.csv",
    prices_path: str = PRICES_PATH,
    out_dir: str = "outputs",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    End-to-end back-test. Reads regime labels + prices, simulates every
    strategy, writes CSVs and PNGs, returns the metrics DataFrame.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load data ----------------------------------------------------------
    regimes = pd.read_csv(regime_labels_path, parse_dates=["date"]).set_index("date").sort_index()
    returns = load_returns(prices_path)

    # Restrict back-test window
    if start_date is None:
        start_date = max(regimes.index.min(), returns.index.min())
    else:
        start_date = pd.to_datetime(start_date)
    if end_date is None:
        end_date = min(regimes.index.max(), returns.index.max())
    else:
        end_date = pd.to_datetime(end_date)

    returns = returns.loc[start_date:end_date]
    regimes = regimes.loc[start_date:end_date]
    daily_idx = returns.index

    # ---- Build weight schedules ---------------------------------------------
    w_rag    = regime_weight_schedule(regimes["pred_label"], daily_idx)
    w_oracle = regime_weight_schedule(regimes["true_label"], daily_idx)
    w_6040   = static_6040_schedule(daily_idx)
    w_spy    = spy_buy_and_hold(daily_idx)
    w_mom    = momentum_weight_schedule(returns)

    # ---- Simulate -----------------------------------------------------------
    strat_returns = pd.DataFrame({
        "RAG-Regime":    simulate(w_rag,    returns),
        "Oracle-Regime": simulate(w_oracle, returns),
        "Static 60/40":  simulate(w_6040,   returns),
        "SPY Buy&Hold":  simulate(w_spy,    returns),
        "Momentum":      simulate(w_mom,    returns),
    })
    # Drop warm-up rows where any strategy has no signal yet
    strat_returns = strat_returns.dropna(how="any").loc[strat_returns.abs().sum(axis=1) > 0]

    # ---- Metrics ------------------------------------------------------------
    weight_map = {
        "RAG-Regime":    w_rag,
        "Oracle-Regime": w_oracle,
        "Static 60/40":  w_6040,
        "SPY Buy&Hold":  w_spy,
        "Momentum":      w_mom,
    }
    metrics_rows = {
        name: compute_metrics(strat_returns[name], weight_map[name]).as_dict()
        for name in strat_returns.columns
    }
    metrics_df = pd.DataFrame(metrics_rows).T

    # ---- Equity & drawdown --------------------------------------------------
    equity = (1 + strat_returns.fillna(0)).cumprod()
    dds = pd.concat({c: drawdown_series(strat_returns[c]) for c in strat_returns.columns}, axis=1)

    # ---- Persist ------------------------------------------------------------
    strat_returns.to_csv(os.path.join(out_dir, "backtest_daily_returns.csv"))
    equity.to_csv(os.path.join(out_dir, "backtest_equity_curves.csv"))
    dds.to_csv(os.path.join(out_dir, "backtest_drawdowns.csv"))
    metrics_df.to_csv(os.path.join(out_dir, "backtest_metrics.csv"))

    plot_equity_curves(equity, os.path.join(out_dir, "equity_curves.png"))
    plot_drawdowns(dds,        os.path.join(out_dir, "drawdowns.png"))
    plot_regime_timeline(regimes[["true_label", "pred_label"]],
                         os.path.join(out_dir, "regime_timeline.png"))

    # ---- Console summary ----------------------------------------------------
    print(f"[backtest] window: {strat_returns.index.min().date()} → {strat_returns.index.max().date()}")
    print(f"[backtest] {len(strat_returns)} trading days, {len(regimes)} rebalances\n")
    print("[backtest] performance summary:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(metrics_df.to_string())

    return metrics_df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="outputs/regime_labels.csv")
    p.add_argument("--prices", default=PRICES_PATH)
    p.add_argument("--out",    default="outputs")
    p.add_argument("--start",  default=None)
    p.add_argument("--end",    default=None)
    args = p.parse_args()

    run_backtest(
        regime_labels_path=args.labels,
        prices_path=args.prices,
        out_dir=args.out,
        start_date=args.start,
        end_date=args.end,
    )
