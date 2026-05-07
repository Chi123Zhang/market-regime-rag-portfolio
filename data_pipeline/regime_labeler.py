"""
regime_labeler.py
=================
Apply *multiple* rule-based heuristics to label each market window.

Why multiple heuristics?
------------------------
The proposal explicitly addresses regime-label ambiguity as a mitigation:
    "Apply to the data several labeling heuristics (VIX thresholds, drawdown
     rules, NBER dates) and calculate the inter-method agreement."

So we don't claim any single label is ground truth. Instead, we attach four
independent labels to each window and report agreement as a quality signal.
The downstream evaluation script can choose its preferred ground truth (or
ensemble them).

Regime taxonomy (from the proposal):
    bull       — uptrend, normal vol
    bear       — sustained drawdown
    high_vol   — elevated volatility regardless of direction
    risk_off   — flight to safety (rare, often overlaps with bear/high_vol)

Public API
----------
    label_windows(windows: pd.DataFrame) -> pd.DataFrame
        Returns the input frame with four added columns:
            label_drawdown, label_vix, label_nber, label_credit
        plus a derived `label_consensus` (modal vote across the four).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import REGIME_THRESHOLDS

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Individual rule functions — each returns a single string label per row
# --------------------------------------------------------------------------
def _label_by_drawdown(row: pd.Series) -> str:
    """
    Drawdown rule:
        bear     if SPY is >=20% below trailing 1y high
        bull     if 200d MA is up AND 63d return > 0
        high_vol if neither, but realised vol > 25%
        else     bull (default)
    """
    if pd.notna(row["spy_dd_from_1y_high"]) and \
       row["spy_dd_from_1y_high"] <= REGIME_THRESHOLDS["bear_drawdown"]:
        return "bear"
    if pd.notna(row["spy_vol_21d"]) and \
       row["spy_vol_21d"] > REGIME_THRESHOLDS["high_vol_realised"]:
        return "high_vol"
    if pd.notna(row["spy_above_200ma"]) and row["spy_above_200ma"] == 1.0 \
       and pd.notna(row["spy_ret_63d"]) and row["spy_ret_63d"] > 0:
        return "bull"
    return "bull"   # benign default — most days historically are uptrend days


def _label_by_vix(row: pd.Series) -> str:
    """
    VIX rule — purely volatility-based.
        high_vol if VIX > 30
        risk_off if VIX > 40 AND credit spread also wide
        bull / bear otherwise based on 63d return sign
    """
    vix = row.get("vix_last", np.nan)
    if pd.notna(vix) and vix > 40 and \
       pd.notna(row.get("credit_spread_oas")) and \
       row["credit_spread_oas"] > REGIME_THRESHOLDS["risk_off_credit_spread"]:
        return "risk_off"
    if pd.notna(vix) and vix > REGIME_THRESHOLDS["high_vol_vix"]:
        return "high_vol"
    if pd.notna(row["spy_ret_63d"]) and row["spy_ret_63d"] < 0:
        return "bear"
    return "bull"


def _label_by_nber(row: pd.Series) -> str:
    """
    NBER recession rule:
        bear if NBER says recession
        bull otherwise (this rule has very low resolution by design)
    """
    if pd.notna(row.get("nber_recession")) and row["nber_recession"] == 1:
        return "bear"
    return "bull"


def _label_by_credit(row: pd.Series) -> str:
    """
    Credit-spread rule — risk-off when credit is selling off.
        risk_off if credit OAS > threshold AND eq-bond corr negative
        high_vol if OAS > threshold but corr still positive
        bull otherwise
    """
    oas = row.get("credit_spread_oas", np.nan)
    corr = row.get("eq_bd_corr_63d", np.nan)
    if pd.notna(oas) and oas > REGIME_THRESHOLDS["risk_off_credit_spread"]:
        if pd.notna(corr) and corr < 0:
            return "risk_off"
        return "high_vol"
    return "bull"


# --------------------------------------------------------------------------
# Consensus
# --------------------------------------------------------------------------
def _consensus(row: pd.Series) -> str:
    """
    Modal vote across the four rule labels. Ties broken by severity:
        risk_off > bear > high_vol > bull
    """
    labels = [row["label_drawdown"], row["label_vix"],
              row["label_nber"], row["label_credit"]]
    counts = pd.Series(labels).value_counts()
    top = counts[counts == counts.max()].index.tolist()
    severity = {"risk_off": 3, "bear": 2, "high_vol": 1, "bull": 0}
    return max(top, key=lambda x: severity.get(x, -1))


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def label_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """
    Attach four heuristic labels + consensus to every window.

    Parameters
    ----------
    windows : pd.DataFrame
        Output of market_windows.build_windows().

    Returns
    -------
    pd.DataFrame
        Same frame with five new columns appended.
    """
    out = windows.copy()
    out["label_drawdown"] = out.apply(_label_by_drawdown, axis=1)
    out["label_vix"]      = out.apply(_label_by_vix,      axis=1)
    out["label_nber"]     = out.apply(_label_by_nber,     axis=1)
    out["label_credit"]   = out.apply(_label_by_credit,   axis=1)
    out["label_consensus"] = out.apply(_consensus, axis=1)

    # Quick agreement diagnostic
    agree = (
        (out["label_drawdown"] == out["label_consensus"]).mean(),
        (out["label_vix"]      == out["label_consensus"]).mean(),
        (out["label_nber"]     == out["label_consensus"]).mean(),
        (out["label_credit"]   == out["label_consensus"]).mean(),
    )
    log.info("Per-rule agreement with consensus: drawdown=%.2f vix=%.2f "
             "nber=%.2f credit=%.2f", *agree)
    log.info("Consensus distribution:\n%s",
             out["label_consensus"].value_counts(normalize=True).round(3).to_string())
    return out


if __name__ == "__main__":
    import logging
    from market_windows import build_windows
    from data_loader import load_all
    from config import OUTPUT_DIR

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    panel = load_all()
    windows = build_windows(panel)
    labelled = label_windows(windows)
    out_path = OUTPUT_DIR / "windows_labelled.parquet"
    labelled.to_parquet(out_path)
    print(labelled[["label_drawdown", "label_vix", "label_nber",
                    "label_credit", "label_consensus"]].tail(10))
