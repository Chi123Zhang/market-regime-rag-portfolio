# 📊 Market Regime RAG Portfolio

A temporal-aware Retrieval-Augmented Generation (RAG) system for financial market regime classification and portfolio backtesting.

This project combines historical market retrieval, feature-based regime prediction, and walk-forward portfolio evaluation into a single reproducible workflow.

The main goal of the project is to study whether retrieval-based reasoning can improve regime classification and portfolio allocation while avoiding look-ahead bias.

---

# 🚀 Overview

Traditional market regime models often behave like black boxes and may overfit to dominant market states.  
This project explores whether retrieving historically similar market environments can provide more stable and interpretable regime predictions.

The system follows four main steps:

1. Build rolling historical market windows
2. Retrieve similar historical periods
3. Predict the market regime using a balanced KNN classifier
4. Evaluate the strategy through walk-forward backtesting

The project also includes a Streamlit demo application for interactive inference and visualization.

---

# 🧠 Main Features

## 🔹 Temporal-aware Retrieval

- Retrieval only uses historical data before the query date
- Prevents look-ahead leakage
- Makes the evaluation more realistic

## 🔹 Balanced Regime Prediction

The final prediction is produced using a feature-space balanced KNN classifier.

The system:

- retrieves similar historical windows
- balances retrieved neighbors by regime
- applies inverse-prior correction
- reduces bull-market dominance

## 🔹 Explainable Outputs

For each prediction, the system returns:

- predicted regime
- confidence score
- retrieved historical windows
- supporting evidence
- optional LLM explanation

## 🔹 Portfolio Backtesting

The project compares several strategies:

- RAG-Regime
- Oracle-Regime
- Static 60/40
- SPY Buy & Hold
- Momentum

Performance metrics include:

- total return
- CAGR
- Sharpe ratio
- maximum drawdown
- turnover

## 🔹 Reproducible Workflow

The repository contains:

- preprocessing scripts
- retrieval pipeline
- inference code
- backtesting engine
- saved outputs
- visualization figures
- Streamlit demo

---

# 🏗️ Repository Structure

```text
market-regime-rag-portfolio/
│
├── app.py
├── financial_rag.py
├── regime_inference.py
├── backtest.py
├── run_backtest.py
├── requirements.txt
│
├── data_pipeline/
│   ├── main.py
│   ├── data_loader.py
│   ├── market_windows.py
│   ├── regime_labeler.py
│   └── outputs/
│
├── outputs/
│   ├── backtest_metrics.csv
│   ├── regime_labels.csv
│   ├── equity_curves.png
│   ├── drawdowns.png
│   └── regime_timeline.png
│
├── tests/
│   └── test_basic.py
│
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Chi123Zhang/market-regime-rag-portfolio.git
cd market-regime-rag-portfolio
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Step 1: Build the data pipeline (optional)

```bash
cd data_pipeline
python main.py
cd ..
```

This generates:

```text
data_pipeline/outputs/corpus.jsonl
```

---

## Step 2: Run the backtest

```bash
python run_backtest.py
```

This generates:

- backtest metrics
- regime labels
- equity curves
- drawdown plots
- regime timeline plots

All outputs are saved in:

```text
outputs/
```

---

## Step 3: Launch the Streamlit demo

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🖥️ Demo Interface

The Streamlit application contains four sections.

## 📌 1. Live RAG Demo

- choose a date
- retrieve similar historical windows
- generate a regime prediction
- display supporting evidence

## 📈 2. Backtest Results

- compare strategies
- display equity curves
- display drawdowns
- compare performance metrics

## 🧭 3. Regime Timeline

- compare predicted regimes with ground truth labels
- visualize regime transitions over time

## 🏗️ 4. Architecture

- visualize the pipeline structure
- explain retrieval and inference flow

---

# 🔬 Methodology

## 1. Data Processing

The pipeline constructs rolling historical market windows using market and macroeconomic indicators.

Examples include:

- SPY
- IEF
- GLD
- volatility indicators
- yield curve signals

Rule-based consensus labels are generated for evaluation purposes.

---

## 2. Retrieval Layer

The retrieval system uses:

- TF-IDF vectorization
- cosine similarity
- temporal filtering

Only historical windows before the query date are allowed during retrieval.

The retrieved windows are used as supporting evidence for regime prediction.

---

## 3. Regime Inference

The final regime prediction is produced by a balanced KNN classifier operating in feature space.

The classifier:

- normalizes market features
- balances retrieved neighbors
- reduces majority-class collapse
- improves sensitivity to non-bull regimes

---

## 4. Optional LLM Explanation

An optional OpenAI explanation layer can summarize retrieved evidence and generate structured explanations.

The LLM layer is only used for explanation.

The final prediction still comes from the feature-space balanced KNN classifier.

---

## 5. Evaluation

The evaluation pipeline includes:

- walk-forward prediction
- strategy simulation
- Sharpe ratio comparison
- drawdown analysis
- turnover analysis

---

# 📊 Example Outputs

The repository includes saved outputs from the backtest.

## Backtest Metrics

Saved in:

```text
outputs/backtest_metrics.csv
```

Metrics include:

- Total Return
- CAGR
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Calmar Ratio
- Hit Rate
- Annual Turnover

---

## Equity Curves

Saved in:

```text
outputs/equity_curves.png
```

The figure compares the cumulative performance of all portfolio strategies.

---

## Drawdown Analysis

Saved in:

```text
outputs/drawdowns.png
```

The figure compares the drawdown behavior of each strategy.

---

## Regime Timeline

Saved in:

```text
outputs/regime_timeline.png
```

The figure compares predicted regimes with rule-based ground truth labels.

---

# ✅ Reproducibility

The repository includes:

- preprocessing scripts
- saved outputs
- generated figures
- evaluation scripts
- walk-forward inference logic

To reproduce the main experiment:

```bash
python run_backtest.py
```

---

# ✅ Basic Testing

Basic reproducibility tests are included.

Run:

```bash
pytest tests/test_basic.py
```

The tests verify:

- required output files exist
- metrics files can be loaded
- regime label outputs contain the expected columns

---

# ⚠️ Notes

- `data_pipeline/` performs offline preprocessing
- `app.py` is the online demo interface
- outputs are pre-generated for faster demo performance
- the OpenAI explanation layer is optional

---

# 🛠️ Troubleshooting

## Missing corpus.jsonl

Run:

```bash
cd data_pipeline
python main.py
cd ..
```

---

## Missing output figures or metrics

Run:

```bash
python run_backtest.py
```

---

## Streamlit does not launch

Reinstall dependencies:

```bash
pip install -r requirements.txt
```

---

## OpenAI API key is unavailable

The system still works normally.

The explanation layer is optional, and the final prediction still comes from the balanced KNN classifier.

---

# 🧠 Key Takeaways

- Temporal filtering is important in financial machine learning
- Retrieval-based reasoning improves interpretability
- Balanced KNN helps reduce majority-regime bias
- Explainability provides practical value beyond prediction accuracy

---

# 📌 Future Work

Possible future extensions include:

- cross-encoder reranking
- dense embedding retrieval
- multi-asset allocation
- real-time market integration
- reinforcement learning allocation
- transaction cost modeling

---

# 👤 Author

Chi (Charlie) Zhang  
M.A. Statistics (Machine Learning Track)  
Columbia University

---

# ⭐ Course Information

Developed for:

STAT GR 5293 — Generative AI Using LLMs  
Columbia University
