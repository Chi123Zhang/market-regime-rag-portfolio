"""
regime_inference.py

Walk-forward regime inference using *feature-space* retrieval with per-class
balanced K-NN and inverse-prior correction.

Usage

    python regime_inference.py
        --start 2008-01-01 --end 2025-12-31
        --top_k 8 --alpha 1.0
        --out outputs/regime_labels.csv
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from financial_rag import MarketRegimeRAG, ALLOWED_REGIMES



# Feature engineering


# These are the 14 numeric features we trust to actually carry regime signal.
# They cover: equity return / risk, equity drawdown, volatility regime,
# cross-asset correlation, credit, macro, recession indicator.
FEATURE_KEYS: Tuple[str, ...] = (
    "spy_ret_63d",
    "spy_vol_63d",
    "spy_sharpe_63d",
    "spy_maxdd_63d",
    "spy_ret_21d",
    "spy_vol_21d",
    "spy_dd_from_1y_high",
    "spy_above_200ma",
    "vix_last",
    "vix_21d_change",
    "eq_bd_corr_63d",
    "yield_curve_10y2y",
    "ten_year_yield",
    "nber_recession",
)


def _to_float(x: Any) -> float:
    """Coerce metadata value to float, with NaN for missing/null."""
    if x is None:
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _build_feature_matrix(records: List[Dict[str, Any]]) -> np.ndarray:
    """Stack metadata into an (N x F) float32 matrix."""
    rows = []
    for r in records:
        meta = r.get("metadata", {}) or {}
        rows.append([_to_float(meta.get(k)) for k in FEATURE_KEYS])
    return np.asarray(rows, dtype=np.float32)


def _zscore_normalize(
    X: np.ndarray, train_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score `X` using stats computed only on rows in `train_mask`.
    Returns (X_normalised, mu, sigma). NaNs are imputed to column means
    on the train set, then to zero post-z-score.
    """
    X_train = X[train_mask]
    mu = np.nanmean(X_train, axis=0)
    sigma = np.nanstd(X_train, axis=0)
    sigma[sigma < 1e-9] = 1.0  # guard against zero-variance columns

    # Impute NaN with the train-mean *before* normalisation.
    X_imp = X.copy()
    nan_mask = np.isnan(X_imp)
    if nan_mask.any():
        col_idx = np.where(nan_mask)[1]
        X_imp[nan_mask] = mu[col_idx]

    X_n = (X_imp - mu) / sigma
    # Belt and braces: any residual NaN -> 0
    X_n = np.nan_to_num(X_n, nan=0.0, posinf=0.0, neginf=0.0)
    return X_n.astype(np.float32), mu, sigma



# Per-class balanced K-NN with inverse-prior correction


def _classify_one(
    q: np.ndarray,
    feats: np.ndarray,
    labels: np.ndarray,
    past_mask: np.ndarray,
    classes: Tuple[str, ...],
    top_k: int,
    alpha: float,
) -> Tuple[str, float, Dict[str, float], int]:
    """
    Return (pred_label, confidence, per_class_score_dict, n_used) for one query.

    score_c = mean_top_k_similarity(class=c, past windows) - alpha * log(prior_c)

    Similarity is `-distance` (Euclidean in z-scored feature space).
    `confidence` is softmax(scores)[pred].
    """
    past_idx = np.where(past_mask)[0]
    if past_idx.size == 0:
        return "unknown", 0.0, {}, 0

    past_labels = labels[past_idx]
    past_feats = feats[past_idx]

    # Distances to all past windows (vectorised)
    diff = past_feats - q[None, :]
    dists = np.sqrt((diff * diff).sum(axis=1))           # shape (n_past,)

    n_past = past_idx.size
    priors = {c: float((past_labels == c).sum()) / n_past for c in classes}

    scores: Dict[str, float] = {}
    n_used = 0
    for c in classes:
        cls_mask = past_labels == c
        n_c = int(cls_mask.sum())
        if n_c == 0:
            continue
        cls_dists = dists[cls_mask]
        k = min(top_k, n_c)
        if k < cls_dists.size:
            partition = np.partition(cls_dists, k - 1)[:k]
        else:
            partition = cls_dists
        mean_sim = float(-partition.mean())              # similarity = -distance
        n_used += k

        # Inverse-prior correction. Guard against log(0).
        prior = max(priors[c], 1.0 / n_past)
        scores[c] = mean_sim - alpha * np.log(prior)

    if not scores:
        return "unknown", 0.0, priors, 0

    pred = max(scores.keys(), key=lambda c: scores[c])

    s_arr = np.array(list(scores.values()), dtype=np.float64)
    s_arr = s_arr - s_arr.max()
    probs = np.exp(s_arr) / np.exp(s_arr).sum()
    conf = float(probs[list(scores.keys()).index(pred)])

    return pred, conf, scores, n_used



# Walk-forward driver


def run_walk_forward_inference(
    rag: MarketRegimeRAG,
    start_date: str = "2008-01-01",
    end_date: str = "2025-12-31",
    top_k: int = 8,
    alpha: float = 1.0,
    use_llm: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward feature-KNN regime classifier with class-balanced retrieval.

    Parameters
    ----------
    rag        : MarketRegimeRAG instance (we use rag.records only).
    start_date : Eval period start (inclusive).
    end_date   : Eval period end (inclusive).
    top_k      : Per-class K-NN size (NOT total retrieval -- each class gets K).
    alpha      : Inverse-prior correction strength. 0 = pure K-NN.
                 Larger alpha penalises high-prior classes more aggressively.
                 alpha = 1 is a sensible default for our 88/6/3/<1 split.
    use_llm    : (passthrough -- keeps the LLM path of the original version).

    Returns DataFrame with the same schema the previous version produced:
        true_label, pred_label, confidence, n_retrieved, method
    """
    classes = tuple(ALLOWED_REGIMES)

    record_dates = np.array([np.datetime64(r["date"]) for r in rag.records])
    record_labels = np.array(
        [r.get("label_consensus", "") for r in rag.records]
    )

    eval_mask = (
        (record_dates >= np.datetime64(start_date))
        & (record_dates <= np.datetime64(end_date))
        & np.isin(record_labels, list(ALLOWED_REGIMES))
    )
    eval_idx = np.where(eval_mask)[0]

    # Warm-up = pre-`start_date` records. Used ONLY for z-score statistics.
    warmup_mask = record_dates < np.datetime64(start_date)
    if warmup_mask.sum() < 30:
        cutoff = max(30, int(0.2 * len(rag.records)))
        warmup_mask = np.zeros_like(warmup_mask)
        warmup_mask[:cutoff] = True
        if verbose:
            print(
                f"[regime_inference] short warm-up "
                f"-> using first {cutoff} records for z-score stats"
            )

    if verbose:
        print(
            f"[regime_inference] feature-KNN walk-forward on "
            f"{eval_idx.size} windows ({start_date} -> {end_date}) | "
            f"top_k_per_class={top_k} alpha={alpha} | "
            f"warm-up={int(warmup_mask.sum())} records"
        )

    raw = _build_feature_matrix(rag.records)
    feats, _mu, _sigma = _zscore_normalize(raw, warmup_mask)

    rows = []
    t0 = time.time()
    for local_i, gi in enumerate(eval_idx):
        r = rag.records[gi]
        target_date = record_dates[gi]
        past_mask = record_dates < target_date

        pred, conf, _scores, n_used = _classify_one(
            q=feats[gi],
            feats=feats,
            labels=record_labels,
            past_mask=past_mask,
            classes=classes,
            top_k=top_k,
            alpha=alpha,
        )

        rows.append(
            {
                "date":         pd.to_datetime(r["date"]),
                "window_start": pd.to_datetime(r["window_start"]),
                "true_label":   r["label_consensus"],
                "pred_label":   pred,
                "confidence":   round(float(conf), 4),
                "n_retrieved":  int(n_used),
                "method":       f"feature_knn_balanced_alpha{alpha}",
            }
        )

        if verbose and (local_i + 1) % 200 == 0:
            print(
                f"  ... {local_i + 1}/{eval_idx.size} done "
                f"in {time.time() - t0:.1f}s"
            )

    df = pd.DataFrame(rows).set_index("date").sort_index()
    if verbose:
        print(f"[regime_inference] finished in {time.time() - t0:.1f}s")
    return df



# Diagnostics


def _per_class_recall(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Recall for each true class -- diagnostic for collapse."""
    out = {}
    for c in ALLOWED_REGIMES:
        mask = y_true == c
        n = int(mask.sum())
        if n == 0:
            out[c] = float("nan")
        else:
            out[c] = round(float((y_pred[mask] == c).sum()) / n, 3)
    return out


def summarise(df: pd.DataFrame) -> Dict[str, Any]:
    """Compact diagnostic on the regime-label output."""
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

    valid = df[df["pred_label"].isin(ALLOWED_REGIMES)].copy()
    if valid.empty:
        return {"n_eval": 0}

    return {
        "n_eval":            int(len(valid)),
        "accuracy":          round(accuracy_score(valid["true_label"], valid["pred_label"]), 4),
        "macro_f1":          round(f1_score(valid["true_label"], valid["pred_label"], average="macro"), 4),
        "cohen_kappa":       round(cohen_kappa_score(valid["true_label"], valid["pred_label"]), 4),
        "true_distribution": valid["true_label"].value_counts(normalize=True).round(3).to_dict(),
        "pred_distribution": valid["pred_label"].value_counts(normalize=True).round(3).to_dict(),
        "per_class_recall":  _per_class_recall(valid["true_label"], valid["pred_label"]),
    }



# CLI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data_pipeline/outputs/corpus.jsonl")
    parser.add_argument("--start",  default="2008-01-01")
    parser.add_argument("--end",    default="2025-12-31")
    parser.add_argument("--top_k",  type=int,   default=8,
                        help="Per-class K (NOT total) for balanced retrieval.")
    parser.add_argument("--alpha",  type=float, default=1.0,
                        help="Inverse-prior correction strength. "
                             "alpha=0 disables it.")
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--out",    default="outputs/regime_labels.csv")
    args = parser.parse_args()

    rag = MarketRegimeRAG(args.corpus)

    df = run_walk_forward_inference(
        rag=rag,
        start_date=args.start,
        end_date=args.end,
        top_k=args.top_k,
        alpha=args.alpha,
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
