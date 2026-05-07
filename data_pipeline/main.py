"""
main.py
=======
Run the entire data + market-windows pipeline end-to-end.

    python main.py [--force-refresh]

Stages
------
1. Download / load prices + macro       (data_loader.load_all)
2. Build rolling market windows         (market_windows.build_windows)
3. Apply heuristic regime labels        (regime_labeler.label_windows)
4. Render to text corpus + write JSONL  (corpus_builder.build_corpus / write)

Outputs
-------
    data_cache/prices.parquet            # raw cached prices
    data_cache/macro.parquet             # raw cached macro series
    outputs/windows.parquet              # numeric features only
    outputs/windows_labelled.parquet     # features + 5 label columns
    outputs/corpus.jsonl                 # text + metadata for the RAG team
    outputs/summary.txt                  # human-readable summary stats

Hand-off notes for the RAG teammate
-----------------------------------
- Each line of corpus.jsonl is a self-contained document. Embed `text`.
- Filter by `metadata['window_end']` (or top-level `date`) when querying so
  no future window leaks into a query made at time T.
- The label columns (`label_*`) are NOT part of the embedded text — they're
  reserved for evaluation only.
"""

from __future__ import annotations

import argparse
import json
import logging

from config import OUTPUT_DIR
from data_loader import load_all
from market_windows import build_windows
from regime_labeler import label_windows
from corpus_builder import build_corpus, write_corpus


def main(force_refresh: bool = False) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("pipeline")

    log.info("=== Stage 1: load prices + macro ===")
    panel = load_all(force_refresh=force_refresh)
    log.info("Panel shape: %s, range %s → %s",
             panel.shape, panel.index.min().date(), panel.index.max().date())

    log.info("=== Stage 2: build market windows ===")
    windows = build_windows(panel)
    windows.to_parquet(OUTPUT_DIR / "windows.parquet")

    log.info("=== Stage 3: heuristic regime labels ===")
    labelled = label_windows(windows)
    labelled.to_parquet(OUTPUT_DIR / "windows_labelled.parquet")

    log.info("=== Stage 4: corpus rendering ===")
    records = build_corpus(labelled)
    write_corpus(records, OUTPUT_DIR / "corpus.jsonl")

    # Human-readable summary
    summary_path = OUTPUT_DIR / "summary.txt"
    with summary_path.open("w") as f:
        f.write("Market-windows pipeline summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Panel rows:     {len(panel)}\n")
        f.write(f"Panel range:    {panel.index.min().date()} → "
                f"{panel.index.max().date()}\n")
        f.write(f"Windows built:  {len(windows)}\n")
        f.write(f"Window step:    every {5} trading days "
                f"(see config.WINDOW_STEP)\n\n")
        f.write("Consensus regime distribution:\n")
        f.write(labelled["label_consensus"].value_counts(normalize=True)
                .round(3).to_string())
        f.write("\n\nSample corpus record (truncated):\n")
        f.write(json.dumps(records[len(records) // 2], indent=2)[:600] + "...\n")
    log.info("Summary written → %s", summary_path)
    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data+windows pipeline.")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-download data even if cached.")
    args = parser.parse_args()
    main(force_refresh=args.force_refresh)
