"""
corpus_builder.py
=================
Convert each numeric market window into a short markdown document that the
RAG vector store will index.

Why text and not just numbers?
------------------------------
Embedding models work on text. We render each window into a deterministic
template that:
  - mentions the date *prominently* (so the temporal-filter teammate can
    extract it cleanly from metadata)
  - contains every feature as both a label and a number (so semantic search
    on phrases like "high volatility risk-off period" still hits the right
    windows)
  - keeps wording neutral (no opinion words like "scary", "crash") so the
    LLM's regime classifier isn't biased by the corpus prose.

Each emitted record has the schema:

    {
        "doc_id":     "window_2008-09-26",
        "date":       "2008-09-26",
        "window_start": "2008-06-27",
        "text":       "<markdown snippet>",
        "metadata":   { ... all numeric features ... },
        "label_consensus": "bear",     # if labels were attached
    }

Public API
----------
    build_corpus(labelled_windows: pd.DataFrame) -> list[dict]
    write_corpus(records, out_path)

The downstream RAG teammate consumes the JSONL output directly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config import CORPUS_TEXT_VERSION

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Number formatting helpers — keep all snippets human-and-LLM-readable.
# --------------------------------------------------------------------------
def _pct(x: float | None, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x*100:.{digits}f}%"


def _num(x: float | None, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.{digits}f}"


# --------------------------------------------------------------------------
# Templating
# --------------------------------------------------------------------------
def _render_window(row: pd.Series) -> str:
    """
    Render one window as a markdown block. Keep this deterministic — the
    same row should always produce the same string (so embedding caches work).
    """
    end   = row.name.strftime("%Y-%m-%d")
    start = pd.Timestamp(row["window_start"]).strftime("%Y-%m-%d")

    text = f"""## Market window {start} → {end}

**Equity (S&P 500 / SPY)**
- 63-day annualised return: {_pct(row['spy_ret_63d'])}
- 63-day annualised volatility: {_pct(row['spy_vol_63d'])}
- 63-day Sharpe: {_num(row['spy_sharpe_63d'])}
- Max drawdown inside window: {_pct(row['spy_maxdd_63d'])}
- 21-day annualised return: {_pct(row['spy_ret_21d'])}
- 21-day annualised volatility: {_pct(row['spy_vol_21d'])}
- Drawdown from trailing 1-year high: {_pct(row['spy_dd_from_1y_high'])}
- Above 200-day moving average: {"yes" if row.get('spy_above_200ma') == 1 else "no"}

**Volatility (VIX)**
- VIX level at window end: {_num(row['vix_last'])}
- VIX change over 21 days: {_num(row['vix_21d_change'])}

**Cross-asset**
- 63-day equity-bond return correlation (SPY vs AGG): {_num(row['eq_bd_corr_63d'])}
- 21-day credit proxy change (log HYG/LQD): {_num(row['credit_proxy_chg_21d'], 3)}

**Macro snapshot (point-in-time)**
- 10y-2y Treasury spread: {_num(row['yield_curve_10y2y'])} pp
- 10y Treasury yield: {_num(row['ten_year_yield'])}%
- Fed funds rate: {_num(row['fed_funds'])}%
- Investment-grade credit OAS: {_num(row['credit_spread_oas'])}%
- CPI YoY: {_num(row['cpi_yoy'])}%
- Unemployment rate: {_num(row['unrate'])}%
- NBER recession indicator: {"on" if row.get('nber_recession') == 1 else "off"}
""".strip()
    return text


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def build_corpus(labelled_windows: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Turn the labelled-windows frame into a list of corpus records ready for
    embedding/indexing.
    """
    records: list[dict[str, Any]] = []
    for end_date, row in labelled_windows.iterrows():
        doc_id = f"window_{end_date.strftime('%Y-%m-%d')}"

        # Metadata = all numeric columns (NaN -> None for clean JSON)
        meta = {}
        for col, val in row.items():
            if col.startswith("label_") or col == "window_start":
                continue
            if pd.isna(val):
                meta[col] = None
            elif isinstance(val, (pd.Timestamp,)):
                meta[col] = val.strftime("%Y-%m-%d")
            else:
                meta[col] = float(val) if not isinstance(val, str) else val

        records.append({
            "doc_id":          doc_id,
            "date":            end_date.strftime("%Y-%m-%d"),
            "window_start":    pd.Timestamp(row["window_start"]).strftime("%Y-%m-%d"),
            "text":            _render_window(row),
            "metadata":        meta,
            "label_drawdown":  row.get("label_drawdown"),
            "label_vix":       row.get("label_vix"),
            "label_nber":      row.get("label_nber"),
            "label_credit":    row.get("label_credit"),
            "label_consensus": row.get("label_consensus"),
            "version":         CORPUS_TEXT_VERSION,
        })

    log.info("Built corpus of %d records (version=%s)",
             len(records), CORPUS_TEXT_VERSION)
    return records


def write_corpus(records: list[dict], out_path: Path) -> None:
    """Write JSONL — one record per line, UTF-8."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d records → %s", len(records), out_path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    from data_loader import load_all
    from market_windows import build_windows
    from regime_labeler import label_windows
    from config import OUTPUT_DIR

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    panel    = load_all()
    windows  = build_windows(panel)
    labelled = label_windows(windows)
    records  = build_corpus(labelled)
    write_corpus(records, OUTPUT_DIR / "corpus.jsonl")

    # Print a sample so the user can see what a record looks like
    print("\nSample record:")
    print(json.dumps(records[len(records) // 2], indent=2)[:1000], "...")
