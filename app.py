import os
import pandas as pd
import streamlit as st

from financial_rag import MarketRegimeRAG
from regime_inference import run_walk_forward_inference, FEATURE_KEYS


# =========================
# Config
# =========================

st.set_page_config(
    page_title="Market Regime RAG Demo",
    layout="wide",
)

CORPUS_PATH = "data_pipeline/outputs/corpus.jsonl"
OUTPUT_DIR = "outputs"

REGIME_LABELS_PATH = os.path.join(OUTPUT_DIR, "regime_labels.csv")
BACKTEST_METRICS_PATH = os.path.join(OUTPUT_DIR, "backtest_metrics.csv")
EQUITY_CURVES_PATH = os.path.join(OUTPUT_DIR, "equity_curves.png")
DRAWDOWNS_PATH = os.path.join(OUTPUT_DIR, "drawdowns.png")
REGIME_TIMELINE_PATH = os.path.join(OUTPUT_DIR, "regime_timeline.png")


# =========================
# Helpers
# =========================

@st.cache_resource
def load_rag():
    return MarketRegimeRAG(CORPUS_PATH)


def find_nearest_record(rag, query_date):
    q_date = pd.to_datetime(query_date)
    return min(
        rag.records,
        key=lambda r: abs(pd.to_datetime(r["date"]) - q_date)
    )


def load_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def show_image_if_exists(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing file: {path}")


def format_market_features(record):
    meta = record.get("metadata", {})
    rows = []
    for k in FEATURE_KEYS:
        rows.append({
            "Feature": k,
            "Value": meta.get(k, None)
        })
    return pd.DataFrame(rows)


def regime_portfolio_rationale(regime):
    if regime == "bull":
        return "Suggested allocation: higher equity exposure, such as 70% SPY, 20% bonds, 10% gold."
    if regime == "bear":
        return "Suggested allocation: defensive mix, such as 30% SPY, 50% Treasuries/bonds, 20% gold."
    if regime == "high_vol":
        return "Suggested allocation: balanced risk control, such as 40% SPY, 40% bonds, 20% gold."
    if regime == "risk_off":
        return "Suggested allocation: capital-preservation mix, such as 20% SPY, 60% Treasuries, 20% gold."
    return "Suggested allocation: neutral benchmark, such as 50% SPY, 40% bonds, 10% gold."


# =========================
# Load System
# =========================

st.title("📊 Retrieval-Based Market Regime Analysis Demo")
st.caption(
    "Feature-space balanced KNN regime classifier with temporal walk-forward evaluation."
)

try:
    rag = load_rag()
except Exception as e:
    st.error(f"Failed to load RAG corpus: {e}")
    st.stop()


# =========================
# Sidebar
# =========================

st.sidebar.title("⚙️ Demo Controls")

query_date = st.sidebar.text_input(
    "Query date",
    value="2020-03-20",
    help="Use YYYY-MM-DD format, for example 2020-03-20."
)

top_k = st.sidebar.slider(
    "Per-class top-k",
    min_value=1,
    max_value=15,
    value=8,
    help="In the improved model, this is top-k per class, not total top-k."
)

alpha = st.sidebar.slider(
    "Inverse-prior correction alpha",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Higher alpha penalizes majority classes more strongly."
)

use_llm_explanation = st.sidebar.checkbox(
    "Use OpenAI LLM explanation",
    value=False,
    help=(
        "The final prediction still comes from feature-KNN. "
        "OpenAI only summarizes the retrieved RAG evidence."
    ),
)

rag_evidence_k = st.sidebar.slider(
    "RAG evidence top-k",
    min_value=1,
    max_value=10,
    value=5,
    help="Number of historical market windows retrieved for explanation.",
)

run_button = st.sidebar.button("Run Market Regime Analysis")

st.sidebar.markdown("### Model")
st.sidebar.info(
    "Final prediction: feature-space balanced KNN from `regime_inference.py`. "
    "RAG retrieval provides similar historical windows as evidence, and the "
    "optional OpenAI LLM layer summarizes that evidence without changing the prediction."
)


# =========================
# Tabs
# =========================

tab1, tab2, tab3, tab4 = st.tabs([
    "Live RAG Regime Demo",
    "Backtest Results",
    "Regime Timeline",
    "Project Architecture"
])


# =========================
# Tab 1: Live Demo
# =========================

with tab1:
    st.header("Live Feature-Based Regime Classification")

    st.write(
        "This version uses date-specific market features and the improved "
        "**feature-space balanced KNN** classifier. When the query date changes, "
        "the nearest market window, features, and prediction also change."
    )

    try:
        nearest_record = find_nearest_record(rag, query_date)

        st.markdown("### Selected Market Window")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Requested Date", query_date)
        col_b.metric("Nearest Window End", nearest_record.get("date"))
        col_c.metric("True Label", nearest_record.get("label_consensus"))

        st.markdown("### Date-Specific Market Features")
        feature_df = format_market_features(nearest_record)
        st.dataframe(feature_df, use_container_width=True)

        st.markdown("### Auto-Generated Structured Query")
        structured_query = rag._query_from_record(nearest_record)
        st.code(structured_query, language="text")

        retrieved_windows = rag.retrieve(
            query=structured_query,
            top_k=rag_evidence_k,
            query_date=nearest_record["date"],
            temporal_mode="strict_past",
        )

        if run_button:
            with st.spinner("Running improved feature-KNN regime inference..."):
                df = run_walk_forward_inference(
                    rag=rag,
                    start_date=nearest_record["date"],
                    end_date=nearest_record["date"],
                    top_k=top_k,
                    alpha=alpha,
                    use_llm=False,
                    verbose=False,
                )

            if df.empty:
                st.warning("No prediction returned for this date.")
            else:
                pred_row = df.iloc[0]

                st.markdown("## Prediction Result")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Predicted Regime", pred_row["pred_label"])
                col2.metric("True Label", pred_row["true_label"])
                col3.metric("Confidence", f"{float(pred_row['confidence']):.3f}")
                col4.metric("Method", pred_row["method"])

                st.markdown("## Grounded Explanation")
                st.write(
                    f"For the selected date, the improved classifier compares the market "
                    f"feature vector against past market windows only. It applies balanced "
                    f"per-class KNN and inverse-prior correction to reduce bull-regime dominance. "
                    f"The predicted regime is **{pred_row['pred_label']}**, while the rule-based "
                    f"reference label is **{pred_row['true_label']}**."
                )

                st.markdown("## RAG Evidence: Similar Historical Windows")

                if retrieved_windows:
                    evidence_rows = []
                    for r in retrieved_windows:
                        evidence_rows.append({
                            "Window Start": r.window_start,
                            "Window End": r.date,
                            "Retrieved Label": r.label_consensus,
                            "Similarity Score": round(float(r.score), 4),
                        })

                    st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True)

                    with st.expander("View retrieved market-window text evidence"):
                        for i, r in enumerate(retrieved_windows, start=1):
                            st.markdown(f"### Evidence {i}: {r.window_start} to {r.date}")
                            st.write(r.text[:1200])
                else:
                    st.warning("No RAG evidence retrieved for this date.")

                st.markdown("## Optional OpenAI LLM Explanation")

                if use_llm_explanation:
                    with st.spinner("Generating LLM explanation from retrieved RAG evidence..."):
                        llm_output = rag.llm_structured_prediction(
                            query=structured_query,
                            retrieved=retrieved_windows,
                        )

                    st.markdown("### 🧠 LLM Interpretation (Explanation Layer)")

                    col_llm_1, col_llm_2, col_llm_3 = st.columns(3)
                    col_llm_1.metric("Final Prediction (KNN)", pred_row["pred_label"])
                    col_llm_2.metric("Rule-Based Reference Label", pred_row["true_label"])
                    col_llm_3.metric("LLM Evidence Interpretation", llm_output.get("regime", "N/A"))

                    if llm_output.get("regime") != pred_row["pred_label"]:
                        st.warning(
                            "The LLM interpretation differs from the final KNN prediction. "
                            "This is expected sometimes because the LLM only summarizes retrieved "
                            "text evidence and does not control the final regime decision."
                        )
                    else:
                        st.success(
                            "The LLM interpretation is consistent with the final KNN prediction."
                        )

                    st.caption(
                        "Important: the final regime prediction above comes from feature-space "
                        "balanced KNN. The LLM only summarizes retrieved historical evidence and "
                        "does not overwrite the prediction."
                    )

                    st.markdown("### 📖 LLM Explanation")
                    st.write(llm_output.get("explanation", "No explanation provided."))

                    st.markdown("### 📊 LLM Supporting Evidence")
                    evidence_list = llm_output.get("evidence", [])

                    if evidence_list:
                        evidence_df = pd.DataFrame(evidence_list)
                        st.dataframe(evidence_df, use_container_width=True)
                    else:
                        st.info("No structured evidence returned by the LLM.")

                    st.markdown("### 💼 LLM Portfolio Suggestion")
                    st.info(llm_output.get("portfolio_rationale", "No portfolio suggestion returned."))

                    with st.expander("View raw LLM JSON output"):
                        st.json(llm_output)

                else:
                    st.info(
                        "OpenAI LLM explanation is turned off. The final prediction still "
                        "comes from feature-space balanced KNN, with RAG evidence shown above."
                    )

                st.markdown("## Portfolio Rationale")
                st.info(regime_portfolio_rationale(pred_row["pred_label"]))

                st.markdown("## Why This Is Different from the Baseline")
                st.success(
                    "The original baseline used TF-IDF retrieval plus majority voting, which often "
                    "collapsed to the bull regime because the corpus was highly imbalanced. "
                    "This improved version uses numeric market features and class-balanced scoring, "
                    "so different query dates can produce different predictions."
                )
        else:
            st.info("Set a query date and click **Run Market Regime Analysis**.")

    except Exception as e:
        st.error(f"Live demo failed: {e}")


# =========================
# Tab 2: Backtest
# =========================

with tab2:
    st.header("Portfolio Backtest Evaluation")

    st.write(
        "This section evaluates whether regime-conditioned allocation improves "
        "risk-adjusted performance compared with benchmark strategies."
    )

    st.markdown("## Backtest Metrics")

    metrics_df = load_csv_if_exists(BACKTEST_METRICS_PATH)

    if metrics_df is not None:
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.warning("Backtest metrics not found. Run `python run_backtest.py` first.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## Equity Curves")
        show_image_if_exists(EQUITY_CURVES_PATH, "Equity curves")

    with col2:
        st.markdown("## Drawdowns")
        show_image_if_exists(DRAWDOWNS_PATH, "Drawdowns")

    st.markdown("### Interpretation")
    st.write(
        "The backtest compares the regime-conditioned strategy with an oracle regime strategy, "
        "a static 60/40 portfolio, SPY buy-and-hold, and a momentum baseline. "
        "The oracle strategy provides an upper bound, while the improved classifier tests whether "
        "feature-based regime inference can close the gap."
    )


# =========================
# Tab 3: Regime Timeline
# =========================

with tab3:
    st.header("Predicted vs Rule-Based Regime Timeline")

    show_image_if_exists(REGIME_TIMELINE_PATH, "Rule-based consensus vs predicted regime")

    labels_df = load_csv_if_exists(REGIME_LABELS_PATH)

    if labels_df is not None:
        st.markdown("## Recent Regime Predictions")
        st.dataframe(labels_df.tail(30), use_container_width=True)

        if "pred_label" in labels_df.columns:
            st.markdown("## Predicted Regime Distribution")
            st.bar_chart(labels_df["pred_label"].value_counts())

        if {"true_label", "pred_label"}.issubset(labels_df.columns):
            st.markdown("## Diagnostic Distribution")

            col1, col2 = st.columns(2)

            with col1:
                st.write("True Label Distribution")
                st.dataframe(
                    labels_df["true_label"].value_counts(normalize=True).round(3)
                )

            with col2:
                st.write("Predicted Label Distribution")
                st.dataframe(
                    labels_df["pred_label"].value_counts(normalize=True).round(3)
                )

            st.info(
                "If the predicted distribution is less concentrated than the baseline, "
                "the improved classifier is reducing majority-class collapse."
            )
    else:
        st.warning("Regime labels not found. Run `python run_backtest.py` first.")


# =========================
# Tab 4: Architecture
# =========================

with tab4:
    st.header("System Architecture")

    st.markdown("## End-to-End Pipeline")

    st.code(
        """
Offline Data Pipeline
├── Download market and macro data
├── Build rolling market windows
├── Generate heuristic regime labels
└── Build corpus.jsonl for downstream inference

Improved Live Inference
├── Select query date
├── Find nearest market window
├── Extract numeric market features
├── Compare against past windows only
├── Apply feature-space balanced KNN
├── Apply inverse-prior correction
├── Retrieve similar historical windows as RAG evidence
├── Optionally summarize evidence with OpenAI LLM
└── Output regime + confidence + evidence + portfolio rationale

Evaluation Layer
├── Walk-forward regime inference
├── Portfolio backtesting
├── Strategy comparison
└── Equity curve / drawdown / metrics output
        """,
        language="text"
    )

    st.markdown("## Why the Improved Version Matters")

    st.write(
        "The original TF-IDF RAG baseline often collapsed to the majority bull regime. "
        "The improved version uses numeric market features and class-balanced KNN for the "
        "final prediction, while RAG retrieval is used to provide similar historical windows "
        "as interpretable evidence. The optional OpenAI LLM layer summarizes this evidence "
        "without changing the model prediction."
    )

    st.success(
        "Key design principle: keep temporal integrity, reduce majority-class bias, "
        "and make regime predictions more responsive to date-specific market conditions."
    )
        
