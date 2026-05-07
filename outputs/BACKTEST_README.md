# Regime Labels + Back-Test Module

> Stage 4–5 of the proposal: **Regime Classifier** (walk-forward inference) +
> **Evaluation Report** (portfolio back-test 2008–2025).
>
> This module consumes `data_pipeline/outputs/corpus.jsonl` produced by
> Ziyi's data module and the existing `financial_rag.py` retriever.

---

## What this module produces

Running `python run_backtest.py` writes the full set of deliverables to `outputs/`:

| File | What it is |
|---|---|
| `regime_labels.csv` | Walk-forward RAG regime prediction per window-end date. **The "regime labels" deliverable.** Columns: `date, window_start, true_label, pred_label, confidence, n_retrieved, method`. |
| `backtest_metrics.csv` | Summary table — total return, CAGR, vol, Sharpe, max drawdown, Calmar, hit rate, turnover — for every strategy. **The headline RQ3 result.** |
| `backtest_daily_returns.csv` | Daily returns of every strategy (one column per strategy). |
| `backtest_equity_curves.csv` | Cumulative-return NAV curves starting at 1.0. |
| `backtest_drawdowns.csv` | Drawdown series per strategy. |
| `equity_curves.png` | Log-scale NAV chart of all strategies (for the report). |
| `drawdowns.png` | Drawdown chart (for the report). |
| `regime_timeline.png` | Predicted vs ground-truth regime over time (for the report). |

---

## Files added in this module

```
market-regime-rag/
├── regime_inference.py     ← walk-forward RAG → regime_labels.csv
├── backtest.py             ← portfolio engine, metrics, plots
└── run_backtest.py         ← orchestrator: regime_inference → backtest
```

Nothing in `data_pipeline/`, `financial_rag.py`, `llm.py`, or `app.py` is modified.

---

## How it works

### 1. Walk-forward regime inference (`regime_inference.py`)

For every window in the back-test period (2008-01-01 → 2025-12-31, ~900 windows):

1. Build a structured query from the window's metadata (SPY return, vol, VIX, drawdown, credit spread, yield curve).
2. Retrieve the top-k most-similar past windows with `date < t` — the strict-past temporal mask comes straight from `financial_rag.py`.
3. In the improved version, the final regime prediction is produced by a feature-space balanced KNN classifier, while the retrieved windows provide supporting evidence; confidence reflects the strength of similarity-based voting.
4. Optionally call the LLM via `--use_llm` (requires `OPENAI_API_KEY`).

The original `financial_rag.MarketRegimeRAG.evaluate_walk_forward` calls the per-window `retrieve()` method, which re-parses every record's date string per query. That dominates runtime (~10 minutes for 900 windows). `regime_inference.run_walk_forward_inference` reuses the *same* TF-IDF matrix but vectorises the retrieval — one `vectorizer.transform` call, one `cosine_similarity` matrix product, then a NumPy mask per row. ~1 second for the same 900 windows. Output is identical.

### 2. Portfolio back-test (`backtest.py`)

Universe: SPY (equity) / IEF (Treasuries) / GLD (gold). Daily prices come from `data_pipeline/data_cache/prices.parquet`.

Regime → weight map (also matches `financial_rag.portfolio_rationale`):

| Regime | SPY | IEF | GLD |
|---|---|---|---|
| `bull`     | 0.70 | 0.20 | 0.10 |
| `bear`     | 0.30 | 0.50 | 0.20 |
| `high_vol` | 0.40 | 0.40 | 0.20 |
| `risk_off` | 0.20 | 0.60 | 0.20 |
| `unknown`  | 0.50 | 0.40 | 0.10 |

Strategies compared (RQ3):

| Strategy | What it is |
|---|---|
| **RAG-Regime**    | Weights set by RAG-predicted regime each rebalance |
| **Oracle-Regime** | Same weight map, but using the rule-based consensus label — *upper bound* for any regime-conditioned strategy with this weight map |
| **Static 60/40**  | Unconditional 60% SPY / 40% IEF (the proposal's "unconditional allocation" baseline) |
| **SPY Buy&Hold**  | Passive equity benchmark |
| **Momentum**      | Long SPY when 21-day SPY return > 0, else IEF (the proposal's "momentum-based alternative") |

**No look-ahead in the back-test itself.** Weights set at close of date `t` are applied to returns from `t+1` — i.e., the daily weight schedule is `.ffill().shift(1)`. The regime label at date `t` was already leakage-free (only past corpus rows used), so the chain is end-to-end clean.

### 3. Orchestrator (`run_backtest.py`)

Runs both steps, prints a single combined summary block.

---

## Running it

```bash
# Full run (default args = the proposal's 2008-2025 back-test, top-k=5)
python run_backtest.py

# Different window range
python run_backtest.py --start 2010-01-01 --end 2025-12-31 --top_k 7

# Use LLM for the structured prediction (requires OPENAI_API_KEY)
python run_backtest.py --use_llm

# Or run the two stages separately
python regime_inference.py --start 2008-01-01 --end 2025-12-31 --out outputs/regime_labels.csv
python backtest.py --labels outputs/regime_labels.csv --out outputs
```

---

## Result interpretation (read this before the demo)

The current `RAG-Regime` strategy is **essentially a tilted long-equity portfolio** — the rule-based RAG predicts `bull` ~99.9% of the time. That's not a bug, it's a finding:

- The corpus is 88% bull-labelled (two decades dominated by uptrends).
- TF-IDF + modal voting on heavily imbalanced data collapses to majority-class prediction.
- Cohen's κ ≈ 0 confirms this — accuracy looks high (88%) only because of the prior.

The **Oracle-Regime** strategy is the upper bound: same weight map, perfect labels. It improves Sharpe from 0.77 → 1.00 and shrinks max drawdown from -37% → -22%, mostly by sidestepping 2008 and 2020. That gap is exactly the value the LLM classifier needs to capture — and is the motivation for the next phase of the project (replacing modal-vote retrieval with a chain-of-thought LLM that reads the retrieved evidence rather than counting labels).

Talking points for the report / TA discussion:

1. **The framework works end-to-end** — leakage-free retrieval, walk-forward simulation, all five strategies run on identical data.
2. **The simple retrieval baseline reaches parity with 60/40** on Sharpe (0.76 vs 0.77) but provides no real regime tilt.
3. **The Oracle bound proves the regime-conditioning thesis**: if labels are right, the strategy materially outperforms both 60/40 and momentum on risk-adjusted return *and* drawdown.
4. **Next step is the LLM classifier**, not a new strategy — that's where the Oracle-RAG gap should close.

---

## Known limitations

- **Heuristic-driven labels.** Same caveat as the data pipeline: ground truth is rule-based, so RQ1 metrics measure agreement with heuristics, not "true" regime.
- **No transaction costs.** Turnover is reported (annualised one-sided) so the reader can apply their own cost assumption — Momentum's 42x turnover would be wiped out by even a few bps. RAG-Regime's 0.09x turnover means costs are immaterial.
- **GLD inception 2004-11.** Pre-GLD rows are dropped from the price panel; the back-test starts at 2008-01-01 anyway.
- **3-asset universe.** Matches the proposal's in-scope definition (US equity / fixed-income / cash-equivalent). Extending to more assets means just adding tickers to `ASSETS` and `REGIME_WEIGHTS`.
