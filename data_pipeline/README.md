# Data + Market Windows Module

> Part 1 of the **Retrieval-Based Market Regime Analysis** project (STAT GR5293).
> This module is owned by **Ziyi**.
> It produces the indexed corpus that the RAG / regime-classifier teammates consume.

---

## 1. What this module does

The proposal's pipeline diagram has five stages:

```
[1. Data Ingestion] → [2. Vector Store] → [3. RAG Retrieval] → [4. LLM Classifier] → [5. Portfolio Engine]
```

This repo covers **stage 1** plus the *temporal-aware windowing* that the proposal lists as a key novelty. Concretely it:

1. **Downloads** daily prices (Yahoo Finance) and macro series (FRED).
2. **Builds rolling market windows** — every 5 trading days, snapshot the previous ~63 trading days into a numeric feature vector (returns, vol, drawdown, VIX, yield curve, credit spreads, NBER indicator, etc.).
3. **Applies four heuristic regime labels** per window (drawdown rule, VIX rule, NBER rule, credit-spread rule) plus a consensus vote — directly implementing the proposal's mitigation for "regime label ambiguity".
4. **Renders each window into a deterministic markdown document** with metadata, written to `outputs/corpus.jsonl` for the vector-store teammate to embed.

**Hand-off contract:** anything downstream needs is in `outputs/corpus.jsonl`. One line per record, JSON. The teammate building the vector store reads that file and embeds the `text` field; the `date` / `metadata.window_end` field is what they filter on for temporal queries.

---

## 2. Why this design (mapping back to the proposal)

| Proposal element | Where it lives in this module |
|---|---|
| "Temporal-aware retrieval mechanism that screens context windows on market date to avoid information leakage" | `market_windows.py` enforces look-ahead-bias-free feature construction; `tests/test_pipeline.py::test_no_lookahead` proves it bit-exactly. |
| "Apply several labeling heuristics (VIX thresholds, drawdown rules, NBER dates) and calculate the inter-method agreement" | `regime_labeler.py` produces 4 independent labels + consensus, and logs per-rule agreement. |
| "Strict temporal separation: the retrieval corpus and evaluation windows do not overlap; walk-forward validation" | Each corpus record carries a `date` field. The evaluation teammate filters `date < query_t` at retrieval time. The window-step granularity (5 days) keeps splits clean. |
| "Ground retrieval context with structured numerical data; stress output schema validation" | Every numeric feature is exposed both as plain-language text (for embedding) and as a typed metadata dict (for tool-style retrieval). |
| 4-regime taxonomy: bull / bear / high_vol / risk_off | Hard-coded in `regime_labeler.py` thresholds, configurable via `config.REGIME_THRESHOLDS`. |
| Backtest range 2005-2025 | `config.START_DATE = "2003-01-01"` (2-year buffer for 252d look-back features), `END_DATE = "2025-12-31"`. |

---

## 3. Repository layout

```
data_pipeline/
├── README.md                    ← this file
├── requirements.txt
├── config.py                    ← all knobs (tickers, dates, thresholds)
├── data_loader.py               ← yfinance + FRED, parquet cache
├── market_windows.py            ← rolling-window feature engineering (core)
├── regime_labeler.py            ← 4 heuristics + consensus
├── corpus_builder.py            ← window → markdown JSONL
├── main.py                      ← orchestrate all four stages
├── tests/
│   └── test_pipeline.py         ← look-ahead bias check + end-to-end
├── data_cache/                  ← raw downloaded series (parquet)
└── outputs/                     ← deliverables for downstream teammates
    ├── windows.parquet              numeric features only
    ├── windows_labelled.parquet     features + 5 label cols
    ├── corpus.jsonl                 ★ THE HAND-OFF FILE ★
    └── summary.txt                  human-readable run summary
```

---

## 4. Quickstart

```bash
# 1. Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the full pipeline (downloads data on first run, then caches)
python main.py

# 3. Re-run forcing a fresh download (e.g., to extend to a newer date)
python main.py --force-refresh

# 4. Verify no look-ahead bias and that everything works end-to-end
python tests/test_pipeline.py
```

Expected first-run runtime: ~30 seconds for download, <5 seconds for windowing/labelling/corpus. After the first run, `data_cache/*.parquet` makes subsequent runs near-instant.

---

## 5. Each script in detail

### `config.py`
Pure constants. Edit this file to:
- change the asset universe (`TICKERS`),
- swap macro series (`FRED_SERIES`),
- tune window length / step (`WINDOW_LONG`, `WINDOW_SHORT`, `WINDOW_STEP`),
- adjust regime thresholds (`REGIME_THRESHOLDS`).

### `data_loader.py`
- `load_prices()` → adj-close DataFrame (cached in `data_cache/prices.parquet`)
- `load_macro()` → FRED DataFrame, forward-filled to business-day frequency
- `load_all()` → inner join on trading days; this is the canonical input

Forward-fill (never back-fill) ensures that on any trading day `t` we know only the most recently *released* macro number — never a future one.

### `market_windows.py`
The core module. For each window-end date `t`:

```python
history = panel.loc[:t]            # ← the only line that touches dates
long_   = history.iloc[-63:]       # 63d slice for slow features
short_  = history.iloc[-21:]       # 21d slice for recent state
year_   = history.iloc[-252:]      # 252d for drawdown-from-high
```

Every feature helper takes a *slice* — so it's structurally impossible to peek beyond `t`. This is verified by `test_no_lookahead`, which corrupts future data and confirms past windows produce identical outputs (max diff = 0.0).

**Features per window (20 total):**

| Group | Features |
|---|---|
| Equity (long) | `spy_ret_63d`, `spy_vol_63d`, `spy_sharpe_63d`, `spy_maxdd_63d` |
| Equity (short) | `spy_ret_21d`, `spy_vol_21d` |
| Trend / drawdown | `spy_dd_from_1y_high`, `spy_above_200ma` |
| Volatility | `vix_last`, `vix_21d_change` |
| Cross-asset | `eq_bd_corr_63d`, `credit_proxy_chg_21d` |
| Macro | `yield_curve_10y2y`, `ten_year_yield`, `fed_funds`, `credit_spread_oas`, `cpi_yoy`, `unrate`, `nber_recession` |

### `regime_labeler.py`
Four independent rule-based labellers plus a consensus vote:

1. **Drawdown rule** — bear if SPY ≤ −20% from trailing 1y high; high_vol if 21d realised vol > 25%; bull otherwise.
2. **VIX rule** — risk_off if VIX > 40 *and* credit spread wide; high_vol if VIX > 30; sign of 63d return otherwise.
3. **NBER rule** — bear during officially-dated recessions (low resolution but a useful sanity check).
4. **Credit-spread rule** — risk_off when high-yield OAS is wide and equity-bond correlation is negative.

The consensus is the modal label, with ties broken by severity (`risk_off > bear > high_vol > bull`). Per-rule agreement with consensus is logged on every run — that's the inter-method-agreement metric the proposal calls for.

### `corpus_builder.py`
Renders each window into a markdown block like:

```
## Market window 2008-06-27 → 2008-09-26

**Equity (S&P 500 / SPY)**
- 63-day annualised return: -8.4%
- 63-day annualised volatility: 24.1%
- ...

**Volatility (VIX)**
- VIX level at window end: 38.5
- ...
```

…and writes it as a JSONL record:

```json
{
  "doc_id": "window_2008-09-26",
  "date": "2008-09-26",
  "window_start": "2008-06-27",
  "text": "## Market window ... (the markdown above)",
  "metadata": { "spy_ret_63d": -0.084, "vix_last": 38.5, ... },
  "label_consensus": "bear",
  "label_drawdown": "bear",
  "label_vix": "high_vol",
  "label_nber": "bear",
  "label_credit": "risk_off"
}
```

**Why text + metadata both?** The text gets embedded for semantic search ("find me high-volatility risk-off windows"). The metadata is for typed filters and for the LLM's structured prompts ("the window had VIX = 38.5"). Labels are *not* in the embedded text — they're held back for evaluation only, so the LLM classifier never accidentally trains on its own ground truth.

### `main.py`
Stitches all four stages together. Outputs are deterministic — running twice produces the same `corpus.jsonl` byte-for-byte (assuming the same cached source data).

---

## 6. Hand-off to teammates

### To the vector-store teammate
1. Read `outputs/corpus.jsonl` line-by-line.
2. Embed `record["text"]` with your chosen model.
3. Store `record["date"]` and `record["metadata"]` as filterable fields.
4. **Critical:** when answering a query made at simulated time `T`, filter `date < T` *before* similarity search. Otherwise the back-test leaks future information.

### To the regime-classifier teammate
- Use `record["text"]` as the retrieved context.
- Use `record["label_consensus"]` as ground truth for RQ1 evaluation.
- For ablations, also report accuracy against each of `label_drawdown`, `label_vix`, `label_nber`, `label_credit` separately — this is what makes the inter-method-agreement story concrete.

### To the portfolio teammate
- `outputs/windows_labelled.parquet` is what you want. It's a clean DataFrame indexed by window-end date with all features and labels. Join it to forward returns to evaluate strategies.

---

## 7. Known limitations / open follow-ups

- **News data is not yet ingested.** The proposal mentions news as a third modality. Adding it requires a NewsAPI key + chunking logic. Stub interface suggested: extend `corpus_builder.py` to append a `news_summary` field per window once news ingestion is wired up. Tracked as a follow-up because the news pipeline is independent of windowing.
- **Universe is US-only ETFs** — matches the proposal's in-scope definition, but if the team later wants to extend to international markets, just add tickers to `config.TICKERS`.
- **Ground-truth labels are heuristic, not expert-annotated.** This is by design (per the proposal), but means RQ1 evaluation should report agreement *between* heuristics, not just against the consensus.
- **Windows are non-overlapping at step granularity (5 days) but feature spans (63 days) overlap heavily.** This is fine for the LLM corpus but matters for any classifier that treats consecutive windows as i.i.d. — they aren't.

---

## 8. Look-ahead bias guarantee

The single most important property of this module is that **no feature for a window ending at date `t` uses any data from after `t`**. We enforce this by:

1. Always slicing the panel with `panel.loc[:t]` before computing anything, then operating only on the slice.
2. Forward-filling macro series (never back-filling).
3. A regression test (`test_no_lookahead`) that:
   - builds windows on the original panel,
   - corrupts every value after a chosen boundary date,
   - rebuilds windows,
   - asserts that windows ending on or before the boundary produce **bit-identical** features.

Current test status: ✅ `max diff = 0.00e+00`.

---

## 9. TA-feedback log (per project rubric §2.3)

> "When you have team discussions with the TA, the TA will give you suggestions and feedback. If those feedback items are taken into account in your final report or demo, they will count toward your score."

Maintain this log as the project progresses — every TA suggestion that touches data / windows / labels gets a row.

| Date | TA suggestion | Action taken | Commit / file |
|---|---|---|---|
| _(empty — fill after first TA discussion)_ | | | |

---

## 10. Reproducibility checklist

- [x] Pinned dependencies (`requirements.txt`)
- [x] All randomness seeded (synthetic data uses `np.random.default_rng(42)`)
- [x] Caching means re-runs produce identical outputs
- [x] Look-ahead bias unit test
- [x] End-to-end integration test on synthetic data (no network needed)
- [ ] CI workflow (bonus per rubric §7) — add `.github/workflows/test.yml` once the repo is on GitHub
