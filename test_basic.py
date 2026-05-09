import os
import pandas as pd

def test_required_output_files_exist():
    assert os.path.exists("outputs/backtest_metrics.csv")
    assert os.path.exists("outputs/regime_labels.csv")
    assert os.path.exists("outputs/equity_curves.png")
    assert os.path.exists("outputs/drawdowns.png")
    assert os.path.exists("outputs/regime_timeline.png")

def test_backtest_metrics_not_empty():
    df = pd.read_csv("outputs/backtest_metrics.csv")

    assert not df.empty
    assert "Sharpe" in df.columns
    assert "Max Drawdown" in df.columns
    assert "CAGR" in df.columns

def test_regime_labels_schema():
    df = pd.read_csv("outputs/regime_labels.csv")

    required_cols = {
        "date",
        "true_label",
        "pred_label",
        "confidence",
        "n_retrieved",
        "method"
    }

    assert required_cols.issubset(set(df.columns))
