# app.py
# Streamlit demo for Retrieval-Based Market Regime Analysis
# Author: Chi Zhang / STAT GR5293 Project

import os
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

from financial_rag import MarketRegimeRAG, ALLOWED_REGIMES


# =========================
# Path Configuration
# =========================

CORPUS_PATH = "data_pipeline/outputs/corpus.jsonl"
OUTPUT_DIR = "outputs"

REGIME_LABELS_PATH = f"{OUTPUT_DIR}/regime_labels.csv"
BACKTEST_METRICS_PATH = f"{OUTPUT_DIR}/backtest_metrics.csv"

EQUITY_CURVE_PATH = f"{OUTPUT_DIR}/equity_curves.png"
DRAWDOWN_PATH = f"{OUTPUT_DIR}/drawdowns.png"
REGIME_TIMELINE_PATH = f"{OUTPUT_DIR}/regime_timeline.png"


# =========================
# Page Setup
# =========================

st.set_page_config(
    page_title="Market Regime RAG Portfolio Demo",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Retrieval-Based Market Regime Analysis Demo")
st.caption(
    "Temporal-aware RAG system for market regime classification, "
    "explainable portfolio allocation, and backtest evaluation."
)


# =========================
# Utility Functions
# =========================

@st.cache_resource
def load_rag_system(corpus_path: str) -> MarketRegimeRAG:
    return MarketRegimeRAG(corpus_path)


@st.cache_data
def load_regime_labels(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_backtest_metrics(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.columns[0].lower() in ["unnamed: 0", "index", "strategy"]:
        df = df.rename(columns={df.columns[0]: "Strategy"})
    return df


def file_status(path: str) -> str:
    return "✅ Found" if os.path.exists(path) else "❌ Missing"


def render_prediction_card(prediction: Dict[str, Any]) -> None:
    regime = prediction.get("regime", "unknown")
    confidence = prediction.get("confidence", 0.0)
    method = prediction.get("method", "unknown")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Regime", regime)

    with col2:
        try:
            st.metric("Confidence", f"{float(confidence):.2f}")
        except Exception:
            st.metric("Confidence", confidence)

    with col3:
        st.metric("Method", method)

    st.subheader("Grounded Explanation")
    st.write(prediction.get("explanation", "No explanation returned."))

    st.subheader("Portfolio Rationale")
    st.info(prediction.get("portfolio_rationale", "No portfolio rationale returned."))


def render_retrieved_evidence(retrieved: List[Dict[str, Any]]) -> None:
    if not retrieved:
        st.warning("No retrieved historical windows were found.")
        return

    for i, item in enumerate(retrieved, start=1):
        with st.expander(
            f"Evidence {i}: {item.get('window_start')} → {item.get('date')} "
            f"| score={item.get('score', 0):.4f} "
            f"| label={item.get('label_consensus', 'N/A')}"
        ):
            st.markdown("#### Metadata")
            meta = item.get("metadata", {})
            if isinstance(meta, dict) and meta:
                meta_df = pd.DataFrame(
                    [{"feature": k, "value": v} for k, v in meta.items()]
                )
                st.dataframe(meta_df, use_container_width=True)
            else:
                st.write("No metadata available.")

            st.markdown("#### Retrieved Market Window Text")
            st.markdown(item.get("text", ""))


# =========================
# Sidebar
# =========================

st.sidebar.header("⚙️ Demo Controls")

query_date = st.sidebar.text_input(
    "Query date",
    value="2020-03-20",
    help="The system retrieves only historical windows before this date.",
)

query_text = st.sidebar.text_area(
    "Market question / scenario",
    value=(
        "Find historical market windows with similar volatility, drawdown, "
        "credit spread, yield curve, and recession conditions."
    ),
    height=120,
)

top_k = st.sidebar.slider("Top-k retrieved windows", min_value=1, max_value=10, value=5)

use_llm = st.sidebar.checkbox(
    "Use OpenAI LLM if API key is available",
    value=False,
    help="If unchecked, the demo uses the deterministic rule-based RAG fallback.",
)

run_button = st.sidebar.button("Run Market Regime Analysis")


# =========================
# System Status
# =========================

with st.expander("🔍 System Status", expanded=False):
    status_df = pd.DataFrame(
        [
            {"Artifact": "RAG corpus", "Path": CORPUS_PATH, "Status": file_status(CORPUS_PATH)},
            {"Artifact": "Regime labels", "Path": REGIME_LABELS_PATH, "Status": file_status(REGIME_LABELS_PATH)},
            {"Artifact": "Backtest metrics", "Path": BACKTEST_METRICS_PATH, "Status": file_status(BACKTEST_METRICS_PATH)},
            {"Artifact": "Equity curve plot", "Path": EQUITY_CURVE_PATH, "Status": file_status(EQUITY_CURVE_PATH)},
            {"Artifact": "Drawdown plot", "Path": DRAWDOWN_PATH, "Status": file_status(DRAWDOWN_PATH)},
            {"Artifact": "Regime timeline", "Path": REGIME_TIMELINE_PATH, "Status": file_status(REGIME_TIMELINE_PATH)},
        ]
    )
    st.dataframe(status_df, use_container_width=True)


# =========================
# Main Demo
# =========================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Live RAG Regime Demo",
        "2. Backtest Results",
        "3. Regime Timeline",
        "4. Project Architecture",
    ]
)


# =========================
# Tab 1: Live RAG Demo
# =========================

with tab1:
    st.header("Live Temporal RAG Regime Classification")

    if not os.path.exists(CORPUS_PATH):
        st.error(
            f"Cannot find `{CORPUS_PATH}`. "
            "Please make sure data_pipeline has generated corpus.jsonl."
        )
    else:
        rag = load_rag_system(CORPUS_PATH)

        st.markdown(
            """
            This demo uses a **strict temporal filter**: for a query date `t`,
            the retriever only searches market windows before `t`.
            This avoids look-ahead leakage during regime classification.
            """
        )

        if run_button:
            with st.spinner("Running temporal RAG retrieval and regime prediction..."):
                try:
                    result = rag.answer(
                        query=query_text,
                        query_date=query_date,
                        top_k=top_k,
                        use_llm=use_llm,
                    )

                    prediction = result.get("prediction", {})
                    retrieved = result.get("retrieved", [])

                    render_prediction_card(prediction)

                    st.divider()
                    st.subheader("Retrieved Historical Evidence")
                    render_retrieved_evidence(retrieved)

                except Exception as e:
                    st.error(f"Demo failed: {e}")
        else:
            st.info("Set a query date and click **Run Market Regime Analysis**.")


# =========================
# Tab 2: Backtest Results
# =========================

with tab2:
    st.header("Portfolio Backtest Evaluation")

    st.markdown(
        """
        This section evaluates whether regime-conditioned portfolio allocation
        improves risk-adjusted performance compared with baseline strategies.
        """
    )

    metrics_df = load_backtest_metrics(BACKTEST_METRICS_PATH)

    if not metrics_df.empty:
        st.subheader("Backtest Metrics")
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.warning("No backtest metrics found. Expected file: `outputs/backtest_metrics.csv`.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Equity Curves")
        if os.path.exists(EQUITY_CURVE_PATH):
            st.image(EQUITY_CURVE_PATH, use_container_width=True)
        else:
            st.warning("Missing equity curve plot.")

    with col2:
        st.subheader("Drawdowns")
        if os.path.exists(DRAWDOWN_PATH):
            st.image(DRAWDOWN_PATH, use_container_width=True)
        else:
            st.warning("Missing drawdown plot.")

    st.markdown(
        """
        **Interpretation:**  
        The RAG-regime strategy is compared against an oracle regime strategy,
        a static 60/40 portfolio, SPY buy-and-hold, and a momentum baseline.
        The oracle strategy serves as an upper bound showing the potential value
        of accurate regime classification.
        """
    )


# =========================
# Tab 3: Regime Timeline
# =========================

with tab3:
    st.header("Predicted vs Rule-Based Regime Timeline")

    if os.path.exists(REGIME_TIMELINE_PATH):
        st.image(REGIME_TIMELINE_PATH, use_container_width=True)
    else:
        st.warning("Missing regime timeline plot.")

    labels_df = load_regime_labels(REGIME_LABELS_PATH)

    if not labels_df.empty:
        st.subheader("Regime Prediction Table")
        st.dataframe(labels_df.tail(50), use_container_width=True)

        if "pred_label" in labels_df.columns:
            st.subheader("Predicted Regime Distribution")
            dist = labels_df["pred_label"].value_counts().reset_index()
            dist.columns = ["Predicted Regime", "Count"]
            st.bar_chart(dist.set_index("Predicted Regime"))
    else:
        st.warning("No regime labels found. Expected file: `outputs/regime_labels.csv`.")


# =========================
# Tab 4: Architecture
# =========================

with tab4:
    st.header("System Architecture")

    st.markdown(
        """
        ### End-to-End Pipeline

        ```text
        Offline Data Pipeline
        ├── Download market and macro data
        ├── Build rolling market windows
        ├── Generate heuristic regime labels
        └── Build corpus.jsonl for retrieval

        Online Demo / Inference
        ├── Load corpus.jsonl
        ├── Retrieve similar historical windows with temporal filtering
        ├── Predict market regime
        ├── Generate grounded explanation
        └── Recommend portfolio allocation

        Evaluation Layer
        ├── Walk-forward regime inference
        ├── Portfolio backtesting
        ├── Strategy comparison
        └── Equity curve / drawdown / metrics output
        ```
        """
    )

    st.markdown(
        """
        ### Why `data_pipeline` matters

        `data_pipeline` is not the Streamlit demo itself.  
        It is the **offline data preparation layer** that generates the artifacts
        consumed by the RAG and backtest modules.

        The most important outputs are:

        - `data_pipeline/outputs/corpus.jsonl`  
          Used by `financial_rag.py` for retrieval.

        - `data_pipeline/data_cache/prices.parquet`  
          Used by `backtest.py` for portfolio simulation.

        This separation keeps the demo fast because the expensive data preparation
        is already completed before the app runs.
        """
    )

    st.subheader("Recommended Demo Talking Point")
    st.success(
        "We separate offline data construction from online inference. "
        "The data pipeline generates leakage-free market-window documents, "
        "while the app performs temporal retrieval, regime prediction, "
        "and portfolio explanation in real time."
    )
        
