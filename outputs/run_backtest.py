"""
run_backtest.py
===============
One-shot orchestrator for the proposal's "Evaluation Report" deliverable.

Steps:
  1. Run RAG walk-forward inference on every market window in the test period.
     Output: outputs/regime_labels.csv  (the regime-labels deliverable)
  2. Run the portfolio back-test using those labels.
     Output: outputs/backtest_metrics.csv + plots
  3. Print a single summary block that combines both.

Run:
    python run_backtest.py
    python run_backtest.py --start 2010-01-01 --end 2025-12-31 --top_k 7
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from financial_rag import MarketRegimeRAG
from regime_inference import run_walk_forward_inference, summarise
from backtest import run_backtest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data_pipeline/outputs/corpus.jsonl")
    parser.add_argument("--prices", default="data_pipeline/data_cache/prices.parquet")
    parser.add_argument("--out",    default="outputs")
    parser.add_argument("--start",  default="2008-01-01")
    parser.add_argument("--end",    default="2025-12-31")
    parser.add_argument("--top_k",  type=int, default=5)
    parser.add_argument("--use_llm", action="store_true",
                        help="Use OpenAI for structured prediction (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- 1. Regime inference --------------------------------------------------
    print("=" * 70)
    print("STEP 1  Regime label inference (walk-forward, leakage-free retrieval)")
    print("=" * 70)
    rag = MarketRegimeRAG(args.corpus)

    labels_df = run_walk_forward_inference(
        rag=rag,
        start_date=args.start,
        end_date=args.end,
        top_k=args.top_k,
        use_llm=args.use_llm,
    )
    labels_path = os.path.join(args.out, "regime_labels.csv")
    labels_df.to_csv(labels_path)
    print(f"\nwrote {labels_path}  ({len(labels_df)} rows)")

    rq1 = summarise(labels_df)
    print("\n[RQ1] regime-classification metrics (RAG vs rule-based consensus):")
    for k, v in rq1.items():
        print(f"  {k}: {v}")

    # ---- 2. Back-test --------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2  Portfolio back-test (RQ3)")
    print("=" * 70)
    metrics_df = run_backtest(
        regime_labels_path=labels_path,
        prices_path=args.prices,
        out_dir=args.out,
        start_date=args.start,
        end_date=args.end,
    )

    # ---- 3. Combined report --------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nRegime classification (RQ1)")
    print(f"  Accuracy   : {rq1.get('accuracy')}")
    print(f"  Macro F1   : {rq1.get('macro_f1')}")
    print(f"  Cohen Kappa: {rq1.get('cohen_kappa')}")
    print(f"  N windows  : {rq1.get('n_eval')}")

    print("\nPortfolio performance (RQ3)")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(metrics_df.to_string())

    print(f"\nAll deliverables written under: {args.out}/")
    for f in sorted(os.listdir(args.out)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
