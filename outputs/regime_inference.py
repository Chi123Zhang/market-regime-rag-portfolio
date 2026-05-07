"""
regime_inference.py
===================
Walk-forward inference that assigns a regime label to every market window
in a chosen back-test period using the temporal RAG.

This is the "regime labels" deliverable from the proposal:

    Regime Classifier — A LLM-based lookup classifier, which takes retrieved
    context as input and gives a regime label that includes a confidence
    score and a systematic argument history.

For every target window ending at date `t`, we:
  1. Build a structured query from `t`'s metadata.
  2. Retrieve top-k similar past windows with `date < t` (strict temporal cut).
  3. Take the modal consensus label of the retrieved windows as the prediction.
  4. Optionally call an LLM for the structured reasoning trace (off by default
     so the back-test is reproducible without API access).

The output file `outputs/regime_labels.csv` is the input to `backtest.py`.

Usage
-----
    python regime_inference.py
        --start 2008-01-01 --end 2025-12-31 --top_k 5
        --out outputs/regime_labels.csv
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from financial_rag import MarketRegimeRAG, ALLOWED_REGIMES


def _modal_label(labels: list) -> tuple:
    """Modal label among `labels` with severity tie-break + confidence."""
    valid = [l for l in labels if l in ALLOWED_REGIMES]
    if not valid:
        return "unknown", 0.0
    counts = Counter(valid)
    top_count = max(counts.values())
    tied = [l for l, c in counts.items() if c == top_count]
    severity = {"risk_off": 3, "bear": 2, "high_vol": 1, "bull": 0}
    pred = max(tied, key=lambda x: severity.get(x, -1))
    return pred, top_count / len(valid)


def run_walk_forward_inference(
    rag: MarketRegimeRAG,
    start_date: str = "2008-01-01",
    end_date: str = "2025-12-31",
    top_k: int = 5,
    use_llm: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run RAG inference on every corpus window whose date falls in [start, end].

    Returns a DataFrame indexed by window-end date with columns:
        true_label    — rule-based consensus from data_pipeline (RQ1 ground truth)
        pred_label    — RAG prediction
        confidence    — fraction of retrieved windows that voted for pred_label
        n_retrieved   — number of past windows actually retrieved
        method        — which prediction path was used

    Implementation note
    -------------------
    We do *not* call `rag.retrieve()` per query in a Python loop — that path
    re-parses every record's date string per query and dominates runtime. We
    take the same TF-IDF matrix the RAG already built and run retrieval in a
    single vectorised pass. Output is identical to looping `rag.retrieve()`
    with `temporal_mode="strict_past"` (modulo numeric stability).
    """
    # ---- Precompute once ---------------------------------------------------
    record_dates  = np.array([np.datetime64(r["date"]) for r in rag.records])
    record_labels = np.array([r.get("label_consensus", "") for r in rag.records])

    target_mask = (
        (record_dates >= np.datetime64(start_date)) &
        (record_dates <= np.datetime64(end_date)) &
        np.isin(record_labels, list(ALLOWED_REGIMES))
    )
    target_idx = np.where(target_mask)[0]

    if verbose:
        print(f"[regime_inference] running walk-forward on "
              f"{len(target_idx)} windows ({start_date} → {end_date})")

    # The LLM path is per-window and rate-limited, so it stays a Python loop.
    if use_llm:
        return _run_walk_forward_llm(rag, target_idx, top_k, verbose)

    # ---- Vectorised retrieval ---------------------------------------------
    t0 = time.time()
    queries = [rag._query_from_record(rag.records[i]) for i in target_idx]
    q_mat = rag.vectorizer.transform(queries)             # (n_targets x V)
    sim = cosine_similarity(q_mat, rag.matrix)            # (n_targets x N)

    rows = []
    for local_i, global_i in enumerate(target_idx):
        r = rag.records[global_i]
        target_date = record_dates[global_i]

        # strict_past temporal mask
        past_mask = record_dates < target_date
        if not past_mask.any():
            rows.append({
                "date":         pd.to_datetime(r["date"]),
                "window_start": pd.to_datetime(r["window_start"]),
                "true_label":   r["label_consensus"],
                "pred_label":   "unknown",
                "confidence":   0.0,
                "n_retrieved":  0,
                "method":       "rule_based_temporal_rag",
            })
            continue

        scores = sim[local_i].copy()
        scores[~past_mask] = -np.inf                      # mask future
        # top_k indices among past records
        order = np.argpartition(-scores, kth=min(top_k, past_mask.sum()) - 1)[:top_k]
        order = order[np.argsort(-scores[order])]

        retrieved_labels = [record_labels[j] for j in order]
        pred, conf = _modal_label(retrieved_labels)

        rows.append({
            "date":         pd.to_datetime(r["date"]),
            "window_start": pd.to_datetime(r["window_start"]),
            "true_label":   r["label_consensus"],
            "pred_label":   pred,
            "confidence":   round(float(conf), 4),
            "n_retrieved":  int(top_k),
            "method":       "rule_based_temporal_rag",
        })

        if verbose and (local_i + 1) % 200 == 0:
            print(f"  ... {local_i+1}/{len(target_idx)} done "
                  f"in {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows).set_index("date").sort_index()
    if verbose:
        print(f"[regime_inference] finished in {time.time() - t0:.1f}s")
    return df


def _run_walk_forward_llm(rag, target_idx, top_k, verbose):
    """LLM-driven path — slow on purpose, one API call per window."""
    rows = []
    t0 = time.time()
    for i, gi in enumerate(target_idx):
        r = rag.records[gi]
        query = rag._query_from_record(r)
        retrieved = rag.retrieve(query=query, top_k=top_k,
                                 query_date=r["date"], temporal_mode="strict_past")
        pred = rag.llm_structured_prediction(query, retrieved)
        rows.append({
            "date":         pd.to_datetime(r["date"]),
            "window_start": pd.to_datetime(r["window_start"]),
            "true_label":   r["label_consensus"],
            "pred_label":   pred.get("regime", "unknown"),
            "confidence":   pred.get("confidence", 0.0),
            "n_retrieved":  len(retrieved),
            "method":       pred.get("method", "unknown"),
        })
        if verbose and (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{len(target_idx)} done in {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows).set_index("date").sort_index()
    if verbose:
        print(f"[regime_inference] finished in {time.time() - t0:.1f}s")
    return df


def summarise(df: pd.DataFrame) -> Dict[str, Any]:
    """Compact diagnostic on the regime-label output."""
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

    valid = df[df["pred_label"].isin(ALLOWED_REGIMES)].copy()
    if valid.empty:
        return {"n_eval": 0}

    return {
        "n_eval":           int(len(valid)),
        "accuracy":         round(accuracy_score(valid["true_label"], valid["pred_label"]), 4),
        "macro_f1":         round(f1_score(valid["true_label"], valid["pred_label"], average="macro"), 4),
        "cohen_kappa":      round(cohen_kappa_score(valid["true_label"], valid["pred_label"]), 4),
        "true_distribution": valid["true_label"].value_counts(normalize=True).round(3).to_dict(),
        "pred_distribution": valid["pred_label"].value_counts(normalize=True).round(3).to_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data_pipeline/outputs/corpus.jsonl")
    parser.add_argument("--start",  default="2008-01-01")
    parser.add_argument("--end",    default="2025-12-31")
    parser.add_argument("--top_k",  type=int, default=5)
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--out",    default="outputs/regime_labels.csv")
    args = parser.parse_args()

    rag = MarketRegimeRAG(args.corpus)

    df = run_walk_forward_inference(
        rag=rag,
        start_date=args.start,
        end_date=args.end,
        top_k=args.top_k,
        use_llm=args.use_llm,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out)
    print(f"\n[regime_inference] wrote {len(df)} rows -> {args.out}")

    summary = summarise(df)
    print("\n[regime_inference] summary metrics (RQ1):")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
