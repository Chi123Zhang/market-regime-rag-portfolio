"""
config.py
=========
Central configuration for the Data + Market-Windows module.

All tunable knobs live here so the rest of the pipeline reads as pure logic.
Edit DATE_RANGE, TICKERS, FRED_SERIES, or WINDOW_* to change the universe
without touching downstream code.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR    = PROJECT_ROOT / "data_cache"     # raw downloaded series (parquet)
OUTPUT_DIR   = PROJECT_ROOT / "outputs"        # processed windows, corpus, labels
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Date range
# --------------------------------------------------------------------------
# Proposal says back-test 2005-2025. We pull a small buffer at the start so the
# first window has enough history to compute long-window features.
START_DATE = "2003-01-01"
END_DATE   = "2025-12-31"

# --------------------------------------------------------------------------
# Asset universe (Yahoo Finance tickers)
# --------------------------------------------------------------------------
# The proposal restricts scope to US equity, fixed-income, cash. We use ETFs
# as proxies because they have continuous daily data going back to 2003.
TICKERS = {
    "SPY":  "US equities (S&P 500)",
    "AGG":  "US aggregate bonds",
    "TLT":  "US 20+ year treasuries",
    "HYG":  "US high-yield credit",
    "LQD":  "US investment-grade credit",
    "IEF":  "US 7-10 year treasuries",
    "GLD":  "Gold",
    "^VIX": "CBOE volatility index",
}

# --------------------------------------------------------------------------
# Macro series (FRED codes)
# --------------------------------------------------------------------------
# Pulled monthly/daily and forward-filled to daily frequency so we can join
# on every trading day.
FRED_SERIES = {
    "DGS10":    "10-Year Treasury yield",
    "DGS2":     "2-Year Treasury yield",
    "DFF":      "Federal funds effective rate",
    "T10Y2Y":   "10Y-2Y spread (yield-curve slope)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS (credit spread)",
    "CPIAUCSL": "Headline CPI (level, monthly)",
    "UNRATE":   "Unemployment rate (monthly)",
    "USREC":    "NBER recession indicator (monthly, 0/1)",
}

# --------------------------------------------------------------------------
# Market-window construction
# --------------------------------------------------------------------------
# A window is a snapshot of market state ending at `t`, summarising the past
# WINDOW_LONG trading days. We emit one window every WINDOW_STEP trading days.
# These match common practice in regime-detection literature (HMMs typically
# use 60-120d look-backs).
WINDOW_LONG  = 63    # ≈ 3 months — long-horizon features (drawdown, vol regime)
WINDOW_SHORT = 21    # ≈ 1 month  — recent-state features (momentum, recent vol)
WINDOW_STEP  = 5     # emit a window every week (Friday close)

# Annualisation factor for daily returns -> annualised stats
TRADING_DAYS_PER_YEAR = 252

# --------------------------------------------------------------------------
# Heuristic regime-labelling thresholds
# --------------------------------------------------------------------------
# These four heuristics produce a "rule-based" ground-truth label per window.
# Per the proposal's mitigation plan, we apply *several* heuristics and report
# inter-method agreement rather than treating any single rule as truth.
#
# Definitions:
#   bull       : SPY is above 200d MA AND 60d return > 0 AND VIX in low/mid regime
#   bear       : SPY drawdown from 252d high <= -20%
#   high_vol   : 21d realised vol of SPY > 25% annualised  OR  VIX > 30
#   risk_off   : equity-bond correlation flips negative AND credit spread widening
REGIME_THRESHOLDS = {
    "bear_drawdown":         -0.20,   # 20% drawdown from rolling 1y high
    "high_vol_realised":      0.25,   # 25% annualised realised vol (21d)
    "high_vol_vix":           30.0,   # VIX level
    "risk_off_credit_spread": 5.0,    # OAS in % — historically high
}

# --------------------------------------------------------------------------
# Corpus settings
# --------------------------------------------------------------------------
# Each window is rendered to a short markdown snippet. These snippets become
# documents in the FAISS / Chroma vector store (built by a teammate).
# We keep the text deterministic and numeric — the LLM relies on exact figures.
CORPUS_TEXT_VERSION = "v1"   # bump if the templating changes
