import pandas as pd
import streamlit as st

from financial_rag import MarketRegimeRAG


st.set_page_config(
    page_title="Temporal RAG Market Regime System",
    layout="wide",
)

st.title("Temporal RAG Market Regime System")
st.caption(
    "A leakage-aware retrieval system for market regime classification, grounded explanation, and portfolio recommendation."
)


@st.cache_resource
def load_system():
    return MarketRegimeRAG("data_pipeline/outputs/corpus.jsonl")


rag = load_system()


st.sidebar.header("Demo Controls")

query = st.sidebar.text_area(
    "Market query",
    value=(
        "Find similar market conditions and classify the current market regime "
        "based on volatility, drawdown, credit stress, and macro indicators."
    ),
    height=120,
)

query_date = st.sidebar.text_input(
    "Query date for temporal filtering",
    value="2022-12-31",
)

top_k = st.sidebar.slider("Retrieved windows", 3, 10, 5)

use_llm = st.sidebar.checkbox(
    "Use optional LLM structured output",
    value=False,
    help="If OPENAI_API_KEY is unavailable, the app automatically uses rule-based RAG.",
)

run_eval = st.sidebar.checkbox(
    "Run walk-forward evaluation",
    value=True,
)

max_eval = st.sidebar.slider(
    "Evaluation windows",
    min_value=50,
    max_value=500,
    value=200,
    step=50,
)


st.subheader("System Architecture")

st.markdown(
    """
**Pipeline**

`market data → rolling market windows → temporal RAG retrieval → structured regime output → portfolio recommendation → walk-forward evaluation`

**Main research design**

This system uses strict temporal filtering.  
For a query at date `T`, the retriever only uses market windows dated **before T**, which helps prevent look-ahead leakage.
"""
)


if st.button("Run Market Regime Analysis"):
    result = rag.answer(
        query=query,
        query_date=query_date,
        top_k=top_k,
        use_llm=use_llm,
    )

    pred = result["prediction"]
    retrieved = result["retrieved"]

    st.subheader("Structured Regime Output")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Regime", pred.get("regime", "unknown"))
    c2.metric("Confidence", pred.get("confidence", 0.0))
    c3.metric("Method", pred.get("method", "unknown"))

    st.json(pred)

    st.subheader("Grounded Explanation")
    st.write(pred.get("explanation", ""))

    st.subheader("Portfolio Rationale")
    st.write(pred.get("portfolio_rationale", ""))

    st.subheader("Retrieved Market Evidence")

    for i, r in enumerate(retrieved, start=1):
        title = (
            f"Evidence {i}: {r['window_start']} → {r['date']} | "
            f"label={r['label_consensus']} | score={r['score']:.3f}"
        )

        with st.expander(title):
            st.markdown(r["text"])

            st.markdown("**Evaluation labels**")
            st.json({
                "consensus": r["label_consensus"],
                "drawdown": r["label_drawdown"],
                "vix": r["label_vix"],
                "nber": r["label_nber"],
                "credit": r["label_credit"],
            })

    if retrieved:
        st.subheader("Retrieved Regime Distribution")
        dist = pd.Series([r["label_consensus"] for r in retrieved]).value_counts()
        st.bar_chart(dist)

else:
    st.info("Click **Run Market Regime Analysis** to start.")


if run_eval:
    st.divider()
    st.subheader("Walk-Forward Evaluation")

    st.markdown(
        """
This evaluation predicts each target window using only earlier windows.  
Ground truth is the consensus heuristic label from the data pipeline.
"""
    )

    eval_result = rag.evaluate_walk_forward(
        start_date="2008-01-01",
        end_date="2025-12-31",
        top_k=top_k,
        max_eval_windows=max_eval,
    )

    metrics = eval_result["metrics"]
    eval_df = eval_result["results"]

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Accuracy", metrics["accuracy"])
    e2.metric("Macro F1", metrics["macro_f1"])
    e3.metric("Cohen's Kappa", metrics["cohen_kappa"])
    e4.metric("N Eval Windows", metrics["n_eval"])

    st.dataframe(eval_df.tail(30), use_container_width=True)

    if not eval_df.empty:
        st.subheader("Prediction Distribution")
        st.bar_chart(eval_df["pred_label"].value_counts())

        st.subheader("True Label Distribution")
        st.bar_chart(eval_df["true_label"].value_counts())
        
