# market-regime-rag-portfolio

# 📊 Market Regime RAG Portfolio

A **temporal-aware Retrieval-Augmented Generation (RAG) system** for financial market regime classification and portfolio allocation.

This project integrates **data pipeline construction, retrieval-based reasoning, regime inference, and backtesting evaluation** into a unified, reproducible workflow.

---

# 🚀 Overview

This system answers a core question:

> Can we improve market regime classification and portfolio decisions by retrieving historically similar market environments — without introducing look-ahead bias?

To address this, we build an **end-to-end pipeline**:

* Construct historical market windows
* Retrieve similar past environments (RAG)
* Generate regime predictions using a feature-based classifier, supported by retrieved historical evidence and optional LLM explanations
* Evaluate performance via backtesting

---

# 🧠 Key Features

## 🔹 Temporal-aware RAG

* Retrieval strictly uses **past data only**
* Prevents **look-ahead bias**
* Ensures realistic financial modeling

## 🔹 Explainable Predictions

* Outputs:

  * Predicted regime (bull / bear / risk_off / high_vol)
  * Confidence score
  * Retrieved historical evidence
* Supports **interpretability**, not just black-box output

## 🔹 End-to-End Pipeline

* Data → Corpus → Retrieval → Inference → Backtest
* Fully reproducible system

## 🔹 Portfolio Backtesting

* Compares:

  * RAG-based strategy
  * Oracle regime
  * Static 60/40
  * SPY Buy & Hold
  * Momentum

## 🔹 Hybrid Decision + Explanation Architecture

* Final regime prediction is produced by a feature-space balanced KNN classifier
* RAG retrieves similar historical market windows as supporting evidence
* Optional LLM layer summarizes retrieved evidence without overriding predictions

---

# 🏗️ Project Structure

```
market-regime-rag-portfolio/
│
├── app.py                  # Streamlit demo app
├── financial_rag.py        # RAG retrieval + inference
├── regime_inference.py     # Regime prediction logic
├── backtest.py             # Backtesting engine
├── run_backtest.py         # Run full evaluation
├── llm.py                  # Optional LLM interface
├── requirements.txt
│
├── data_pipeline/          # Offline data processing
│   ├── main.py
│   ├── data_loader.py
│   ├── market_windows.py
│   ├── regime_labeler.py
│   └── outputs/            # Generated corpus
│
├── outputs/                # Backtest results
│   ├── regime_labels.csv
│   ├── backtest_metrics.csv
│   ├── equity_curves.png
│   ├── drawdowns.png
│   └── regime_timeline.png
```

---

# ⚙️ Installation

## 1. Clone repository

```
git clone https://github.com/Chi123Zhang/market-regime-rag-portfolio.git
cd market-regime-rag-portfolio
```

## 2. Install dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Step 1: Build data pipeline (if needed)

```
cd data_pipeline
python main.py
cd ..
```

This generates:

```
data_pipeline/outputs/corpus.jsonl
```

---

## Step 2: Run backtest (optional)

```
python run_backtest.py
```

Outputs:

* equity curves
* drawdowns
* performance metrics

---

## Step 3: Launch demo app

```
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

# 🖥️ Demo Interface

The app includes four tabs:

### 📌 1. Live RAG

* Select a date
* Retrieve historical analogs
* Generate regime prediction + explanation

### 📈 2. Backtest Results

* Compare strategies
* View equity curves & drawdowns

### 🧭 3. Regime Timeline

* Ground truth vs predicted regimes

### 🏗️ 4. Architecture

* Visual overview of pipeline

---

# 🔬 Methodology

## 1. Data Processing

* Construct rolling market windows
* Label regimes using rule-based consensus

## 2. Retrieval (RAG)

* BM25 / embedding-based similarity
* Temporal filtering: only past data allowed

## 3. Inference

* Feature-space balanced KNN produces the final regime prediction
* Retrieval (RAG) provides similar historical windows as supporting evidence
* Optional LLM layer generates structured explanations based on retrieved evidence

## 4. Evaluation

* Portfolio allocation based on regime
* Backtesting vs benchmarks

---

# 📊 Example Outputs

* Equity Curve Comparison
* Drawdown Analysis
* Regime Classification Timeline

---

# ⚠️ Notes

* `data_pipeline` is **offline computation**
* `app.py` is **online inference demo**
* Outputs are pre-generated for fast demo

---

# 🧠 Key Insights

* Retrieval-based reasoning can improve regime stability
* Temporal constraints are critical in financial ML
* Explainability adds practical value beyond prediction

---

# 📌 Future Work

* Cross-encoder reranking
* Multi-asset extension
* Real-time data integration
* Reinforcement learning for allocation

---

# 👤 Author

Chi (Charlie) Zhang  
M.A. Statistics (Machine Learning Track), Columbia University

---

# ⭐ If you find this useful

Feel free to star the repo!
