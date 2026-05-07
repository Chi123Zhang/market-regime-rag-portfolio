import os
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


CORPUS_PATH = "data_pipeline/outputs/corpus.jsonl"
ALLOWED_REGIMES = ["bull", "bear", "high_vol", "risk_off"]

def _regime_diverse_local_order(
    scores: np.ndarray,
    candidate_indices: List[int],
    records: List[Dict[str, Any]],
    top_k: int,
    pool_multiplier: int = 10,
) -> np.ndarray:
    """
    Return top-k local indices with light regime diversity.

    This keeps the original similarity ranking as the primary signal, but avoids
    the common failure mode where the retrieved set is entirely dominated by the
    majority regime, usually bull. It first looks inside a reasonably large
    high-similarity candidate pool, selects the best available example from each
    non-duplicate regime, and then fills remaining slots by similarity order.

    Important: this is post-retrieval diversification; it does not change the
    temporal mask or introduce future information.
    """
    if top_k <= 0 or len(scores) == 0:
        return np.array([], dtype=int)

    ranked = list(np.argsort(scores)[::-1])
    pool_size = min(len(ranked), max(top_k * pool_multiplier, top_k))
    pool = ranked[:pool_size]

    selected: List[int] = []
    seen_regimes = set()

    # Prefer less frequent / risk-sensitive regimes first, then bull.
    regime_priority = ["risk_off", "bear", "high_vol", "bull"]

    for regime in regime_priority:
        for local_idx in pool:
            global_idx = candidate_indices[local_idx]
            label = records[global_idx].get("label_consensus", "")
            if label == regime and local_idx not in selected:
                selected.append(local_idx)
                seen_regimes.add(regime)
                break
        if len(selected) >= top_k:
            break

    # Fill remaining slots using the original similarity ranking.
    for local_idx in ranked:
        if len(selected) >= top_k:
            break
        if local_idx not in selected:
            selected.append(local_idx)

    return np.array(selected[:top_k], dtype=int)


@dataclass
class RetrievedWindow:
    score: float
    doc_id: str
    date: str
    window_start: str
    text: str
    metadata: Dict[str, Any]
    label_consensus: str
    label_drawdown: str
    label_vix: str
    label_nber: str
    label_credit: str


class MarketRegimeRAG:
    """
    Temporal-aware RAG system for market regime analysis.

    Core idea:
    1. Each market window is a document.
    2. Retrieval is filtered by time to avoid look-ahead leakage.
    3. Regime classification is grounded in retrieved evidence.
    4. Evaluation compares RAG predictions against rule-based consensus labels.
    """

    def __init__(self, corpus_path: str = CORPUS_PATH):
        self.corpus_path = corpus_path
        self.records = self._load_corpus(corpus_path)
        self.texts = [r["text"] for r in self.records]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=12000,
            min_df=2,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def _load_corpus(self, corpus_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"Cannot find {corpus_path}. Run data_pipeline/main.py first."
            )

        records = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        records = sorted(records, key=lambda x: x["date"])
        return records

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        query_date: Optional[str] = None,
        temporal_mode: str = "strict_past",
    ) -> List[RetrievedWindow]:
        """
        Retrieve relevant windows.

        temporal_mode:
        - strict_past: only use windows before query_date
        - up_to_date: use windows up to and including query_date
        - all: no temporal filtering
        """

        candidate_indices = []

        for i, r in enumerate(self.records):
            if query_date is None or temporal_mode == "all":
                candidate_indices.append(i)
                continue

            record_date = pd.to_datetime(r["date"])
            q_date = pd.to_datetime(query_date)

            if temporal_mode == "strict_past":
                if record_date < q_date:
                    candidate_indices.append(i)
            elif temporal_mode == "up_to_date":
                if record_date <= q_date:
                    candidate_indices.append(i)

        if not candidate_indices:
            return []

        q_vec = self.vectorizer.transform([query])
        sub_matrix = self.matrix[candidate_indices]
        scores = cosine_similarity(q_vec, sub_matrix).flatten()

        order = _regime_diverse_local_order(scores, candidate_indices, self.records, top_k)

        results = []
        for local_idx in order:
            global_idx = candidate_indices[local_idx]
            r = self.records[global_idx]

            results.append(
                RetrievedWindow(
                    score=float(scores[local_idx]),
                    doc_id=r.get("doc_id", ""),
                    date=r.get("date", ""),
                    window_start=r.get("window_start", ""),
                    text=r.get("text", ""),
                    metadata=r.get("metadata", {}),
                    label_consensus=r.get("label_consensus", ""),
                    label_drawdown=r.get("label_drawdown", ""),
                    label_vix=r.get("label_vix", ""),
                    label_nber=r.get("label_nber", ""),
                    label_credit=r.get("label_credit", ""),
                )
            )

        return results

    def build_context(self, retrieved: List[RetrievedWindow]) -> str:
        blocks = []
        for i, r in enumerate(retrieved, start=1):
            block = f"""
[Evidence {i}]
Date: {r.window_start} to {r.date}
Similarity score: {r.score:.4f}
Consensus label for evaluation only: {r.label_consensus}

{r.text}
""".strip()
            blocks.append(block)
        return "\n\n".join(blocks)

    def rule_based_rag_prediction(self, retrieved: List[RetrievedWindow]) -> Dict[str, Any]:
        """
        Non-LLM baseline prediction using retrieved evidence.
        This is useful when OpenAI quota is unavailable.
        """

        if not retrieved:
            return {
                "regime": "unknown",
                "confidence": 0.0,
                "evidence": [],
                "explanation": "No past market windows were available after temporal filtering.",
                "portfolio_rationale": "Use a neutral benchmark allocation because the regime is unknown.",
            }

        labels = [r.label_consensus for r in retrieved if r.label_consensus in ALLOWED_REGIMES]

        if not labels:
            return {
                "regime": "unknown",
                "confidence": 0.0,
                "evidence": [],
                "explanation": "Retrieved windows did not contain usable labels.",
                "portfolio_rationale": "Use a neutral benchmark allocation.",
            }

        counts = pd.Series(labels).value_counts()
        regime = counts.index[0]
        confidence = float(counts.iloc[0] / len(labels))

        evidence = [
            {
                "date": r.date,
                "window_start": r.window_start,
                "label": r.label_consensus,
                "score": round(r.score, 4),
            }
            for r in retrieved[:3]
        ]

        return {
            "regime": regime,
            "confidence": round(confidence, 3),
            "evidence": evidence,
            "explanation": (
                f"The temporal RAG system retrieved {len(retrieved)} past market windows. "
                f"The most frequent retrieved consensus regime is '{regime}', appearing in "
                f"{counts.iloc[0]} of {len(labels)} retrieved windows. This prediction uses only "
                f"windows dated before the query date, reducing look-ahead leakage."
            ),
            "portfolio_rationale": self.portfolio_rationale(regime),
        }

    def llm_structured_prediction(
        self,
        query: str,
        retrieved: List[RetrievedWindow],
        model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """
        Optional LLM layer.
        If OPENAI_API_KEY is not available, automatically falls back to rule-based RAG.
        """

        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key or OpenAI is None:
            pred = self.rule_based_rag_prediction(retrieved)
            pred["method"] = "rule_based_rag_fallback"
            return pred

        context = self.build_context(retrieved)

        prompt = f"""
You are a financial market-regime analysis assistant.

Task:
Classify the market regime using ONLY the retrieved market-window evidence.

Allowed regimes:
- bull
- bear
- high_vol
- risk_off

Important temporal rule:
The retrieved evidence has already been filtered to avoid future information leakage.
Do not use outside knowledge.

User query:
{query}

Retrieved market evidence:
{context}

Return ONLY valid JSON with this schema:
{{
  "regime": "bull | bear | high_vol | risk_off",
  "confidence": 0.0,
  "evidence": [
    {{
      "date": "YYYY-MM-DD",
      "reason": "short evidence-based reason"
    }}
  ],
  "explanation": "grounded explanation based only on retrieved evidence",
  "portfolio_rationale": "portfolio implication based on the predicted regime"
}}
"""

        client = OpenAI(api_key=api_key)

        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful financial RAG assistant. "
                            "Return strict JSON only. Use only retrieved evidence."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            content = response.choices[0].message.content.strip()
            content = re.sub(r"^```json", "", content).strip()
            content = re.sub(r"^```", "", content).strip()
            content = re.sub(r"```$", "", content).strip()

            parsed = json.loads(content)

            if parsed.get("regime") not in ALLOWED_REGIMES:
                raise ValueError("Invalid regime returned by LLM.")

            parsed["method"] = "llm_structured_rag"
            return parsed

        except Exception as e:
            pred = self.rule_based_rag_prediction(retrieved)
            pred["method"] = "rule_based_rag_fallback_after_llm_error"
            pred["llm_error"] = str(e)
            return pred

    def portfolio_rationale(self, regime: str) -> str:
        if regime == "bull":
            return "Suggested allocation: higher equity exposure, such as 70% SPY, 20% bonds, 10% gold."
        if regime == "bear":
            return "Suggested allocation: defensive mix, such as 30% SPY, 50% Treasuries/bonds, 20% gold."
        if regime == "high_vol":
            return "Suggested allocation: balanced risk control, such as 40% SPY, 40% bonds, 20% gold."
        if regime == "risk_off":
            return "Suggested allocation: capital-preservation mix, such as 20% SPY, 60% Treasuries, 20% gold."
        return "Suggested allocation: neutral benchmark, such as 50% SPY, 40% bonds, 10% gold."

    def answer(
        self,
        query: str,
        query_date: str,
        top_k: int = 5,
        use_llm: bool = False,
    ) -> Dict[str, Any]:
        retrieved = self.retrieve(
            query=query,
            top_k=top_k,
            query_date=query_date,
            temporal_mode="strict_past",
        )

        if use_llm:
            prediction = self.llm_structured_prediction(query, retrieved)
        else:
            prediction = self.rule_based_rag_prediction(retrieved)
            prediction["method"] = "rule_based_temporal_rag"

        return {
            "query": query,
            "query_date": query_date,
            "retrieved": [r.__dict__ for r in retrieved],
            "prediction": prediction,
        }

    def evaluate_walk_forward(
        self,
        start_date: str = "2008-01-01",
        end_date: str = "2025-12-31",
        top_k: int = 5,
        max_eval_windows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Walk-forward evaluation:
        For each target window t, predict its regime using only windows before t.
        Ground truth = label_consensus from data_pipeline.
        """

        targets = [
            r for r in self.records
            if pd.to_datetime(start_date) <= pd.to_datetime(r["date"]) <= pd.to_datetime(end_date)
            and r.get("label_consensus") in ALLOWED_REGIMES
        ]

        if max_eval_windows:
            targets = targets[-max_eval_windows:]

        rows = []

        for r in targets:
            query = self._query_from_record(r)
            retrieved = self.retrieve(
                query=query,
                top_k=top_k,
                query_date=r["date"],
                temporal_mode="strict_past",
            )
            pred = self.rule_based_rag_prediction(retrieved)

            rows.append({
                "date": r["date"],
                "true_label": r["label_consensus"],
                "pred_label": pred["regime"],
                "confidence": pred["confidence"],
                "n_retrieved": len(retrieved),
            })

        df = pd.DataFrame(rows)
        valid = df[df["pred_label"].isin(ALLOWED_REGIMES)].copy()

        if valid.empty:
            metrics = {
                "accuracy": None,
                "macro_f1": None,
                "cohen_kappa": None,
                "n_eval": 0,
            }
        else:
            metrics = {
                "accuracy": round(accuracy_score(valid["true_label"], valid["pred_label"]), 4),
                "macro_f1": round(f1_score(valid["true_label"], valid["pred_label"], average="macro"), 4),
                "cohen_kappa": round(cohen_kappa_score(valid["true_label"], valid["pred_label"]), 4),
                "n_eval": int(len(valid)),
            }

        return {
            "metrics": metrics,
            "results": df,
        }

    def _query_from_record(self, r: Dict[str, Any]) -> str:
        meta = r.get("metadata", {})
        return (
            "Find historical market windows with similar equity return, volatility, VIX, "
            "drawdown, credit spread, yield curve, and recession conditions. "
            f"SPY 63-day return {meta.get('spy_ret_63d')}; "
            f"SPY 21-day volatility {meta.get('spy_vol_21d')}; "
            f"VIX {meta.get('vix_last')}; "
            f"drawdown from 1-year high {meta.get('spy_dd_from_1y_high')}; "
            f"credit spread {meta.get('credit_spread_oas')}; "
            f"yield curve {meta.get('yield_curve_10y2y')}."
        )
    
