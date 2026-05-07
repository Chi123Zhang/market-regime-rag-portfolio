"""
test_pipeline.py
================
Two integration tests using synthetic data (so we don't need network access).

1. test_no_lookahead — perturb future data, recompute windows, verify that
   features for any window ending at date `t` are unchanged. This is the
   strongest possible no-look-ahead guarantee.

2. test_pipeline_runs_end_to_end — synthesise a full panel, run all four
   stages, assert every output file exists and has the expected shape.

Run from the project root:
    python -m tests.test_pipeline
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Make project modules importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_windows import build_windows
from regime_labeler import label_windows
from corpus_builder import build_corpus, write_corpus


# --------------------------------------------------------------------------
# Fixture: synthetic price + macro panel
# --------------------------------------------------------------------------
def synth_panel(seed: int = 42, n_days: int = 1500) -> pd.DataFrame:
    """
    Generate a deterministic panel that has the columns market_windows.py
    expects. Returns/vols are reasonable so labels come out diverse.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n_days)

    def gbm(mu=0.06, sigma=0.18, start=100):
        # Geometric Brownian motion, daily
        dt = 1 / 252
        rets = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_days)
        return start * np.exp(np.cumsum(rets))

    panel = pd.DataFrame(index=idx)
    panel.index.name = "date"

    panel["SPY"] = gbm(0.08, 0.18, 300)
    panel["AGG"] = gbm(0.03, 0.05, 100)
    panel["TLT"] = gbm(0.02, 0.13, 90)
    panel["HYG"] = gbm(0.05, 0.10, 80)
    panel["LQD"] = gbm(0.04, 0.07, 110)
    panel["IEF"] = gbm(0.02, 0.06, 100)
    panel["GLD"] = gbm(0.05, 0.16, 130)

    # VIX: mean-reverting around 18, occasional spikes
    vix = np.full(n_days, 18.0)
    for i in range(1, n_days):
        vix[i] = 0.95 * vix[i-1] + 0.05 * 18 + rng.normal(0, 1.5)
        if rng.random() < 0.005:
            vix[i] += rng.uniform(15, 30)   # crisis spike
    panel["^VIX"] = np.clip(vix, 9, 80)

    # Macro
    panel["DGS10"]  = 2.5 + 0.5 * np.sin(np.linspace(0, 6, n_days)) \
                          + rng.normal(0, 0.05, n_days)
    panel["DGS2"]   = 1.5 + 0.4 * np.sin(np.linspace(0, 5, n_days)) \
                          + rng.normal(0, 0.05, n_days)
    panel["DFF"]    = 0.5 + 1.0 * np.sin(np.linspace(0, 4, n_days)) ** 2
    panel["T10Y2Y"] = panel["DGS10"] - panel["DGS2"]
    panel["BAMLH0A0HYM2"] = 4 + 2 * (panel["^VIX"] / 30) + rng.normal(0, 0.3, n_days)
    panel["CPIAUCSL"] = 250 * (1 + 0.025) ** (np.arange(n_days) / 252)
    panel["CPI_YOY"]  = panel["CPIAUCSL"].pct_change(252) * 100
    panel["UNRATE"]   = 5 + rng.normal(0, 0.3, n_days)
    panel["USREC"]    = (panel["^VIX"].rolling(60).mean() > 25).astype(float)

    return panel.dropna()


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_no_lookahead() -> None:
    """
    Build windows, then corrupt the *future* portion of the panel and rebuild.
    Window features for any end-date <= the corruption boundary must be
    identical. If they aren't, we have a look-ahead bug.
    """
    panel = synth_panel()
    boundary = panel.index[1000]      # split the panel ~2/3 in
    base_windows = build_windows(panel)

    corrupted = panel.copy()
    # Multiply every row after the boundary by a wild factor
    corrupted.loc[corrupted.index > boundary, ["SPY", "AGG", "^VIX"]] *= 5.0
    corrupted_windows = build_windows(corrupted)

    # Compare windows that end on or before the boundary
    a = base_windows.loc[base_windows.index <= boundary]
    b = corrupted_windows.loc[corrupted_windows.index <= boundary]
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    numeric_cols = a.select_dtypes(include=[np.number]).columns
    diff = (a[numeric_cols] - b[numeric_cols]).abs().max().max()
    assert diff < 1e-9, (
        f"Look-ahead bias detected — features changed by {diff} when future "
        "data was corrupted."
    )
    print(f"✓ test_no_lookahead passed (max diff = {diff:.2e})")


def test_pipeline_runs_end_to_end(tmp_dir: Path) -> None:
    """
    Run all four stages. Verify shapes and that corpus.jsonl is parseable.
    """
    panel    = synth_panel()
    windows  = build_windows(panel)
    labelled = label_windows(windows)
    records  = build_corpus(labelled)

    # Shape sanity
    assert len(windows) > 50, f"Too few windows: {len(windows)}"
    assert len(records) == len(labelled)
    assert {"text", "metadata", "date", "label_consensus"}.issubset(records[0])

    # Every record's text must mention its window-end date
    for r in records:
        assert r["date"] in r["text"], f"Date missing in text for {r['doc_id']}"

    # Every label is one of the four allowed values
    allowed = {"bull", "bear", "high_vol", "risk_off"}
    for r in records:
        assert r["label_consensus"] in allowed, \
            f"Bad label: {r['label_consensus']}"

    # Write JSONL and re-read it
    out_path = tmp_dir / "corpus.jsonl"
    write_corpus(records, out_path)
    reread = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(reread) == len(records)

    print(f"✓ test_pipeline_runs_end_to_end passed "
          f"({len(windows)} windows, {len(records)} records)")
    print(f"  consensus distribution:\n"
          f"  {labelled['label_consensus'].value_counts().to_dict()}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    test_no_lookahead()
    with tempfile.TemporaryDirectory() as td:
        test_pipeline_runs_end_to_end(Path(td))
    print("\nAll tests passed ✅")
