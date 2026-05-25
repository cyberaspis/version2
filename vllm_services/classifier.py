"""
Vishing classifier combining LLM inference (via vLLM) with heuristic scoring.

Provides:
- VishingClassifier: async classifier using VLLMClient + heuristics
- Metric evaluation: threshold search, confusion matrix, ROC-AUC
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from .vllm_client import VLLMClient
from . import heuristics
from . import prompts


class CallStateManager:
    """
    Manages the persistent risk state of active calls.
    
    Accumulates risk points based on LLM results and heuristics.
    Points are clamped between 0 and 200.
    """
    def __init__(self):
        # call_id -> risk_points
        self._states: Dict[str, float] = {}

    def get_risk(self, call_id: str) -> float:
        return self._states.get(call_id, 0.0)

    def update(self, call_id: str, llm_score: float, heuristic_score: float) -> Tuple[float, str]:
        """
        Update risk points for a call.
        
        Logic:
        - Apply small decay to avoid permanent high scores on early noise.
        - LLM contribution: proportional to (llm_score - 0.5) so it can both increase and decrease.
        - Heuristic boost: smaller than before to reduce false positives.
        """
        current = self._states.get(call_id, 0.0)
        # Decay: allow recovery as more benign segments arrive
        current *= 0.90
        
        # LLM Contribution (centered at 0.5)
        delta = (llm_score - 0.5) * 60.0
            
        # Heuristic Contribution (reduced)
        h_boost = heuristic_score * 20.0
        
        new_total = max(0.0, min(200.0, current + delta + h_boost))
        self._states[call_id] = new_total
        
        status = "SAFE"
        if new_total > 150:
            status = "CRITICAL"
        elif new_total > 100:
            status = "VISHING"
        elif new_total > 50:
            status = "WARNING"
            
        return new_total, status

    def cleanup(self, call_id: str):
        if call_id in self._states:
            del self._states[call_id]


class VishingClassifier:
    """
    Vishing detection classifier.

    Combines logprobs-based YES/NO LLM probability with Greek
    keyword heuristic scoring to produce a final vishing score.
    """

    def __init__(self, client: VLLMClient):
        self.client = client
        self.state_manager = CallStateManager()

    async def classify(self, text: str, call_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify a single text for vishing probability.
        Uses a sliding window of the last 1000 characters for the LLM.

        Returns dict with:
            - score: combined final score (current segment)
            - prob_yes: LLM probability of YES
            - heuristic_score: keyword-based heuristic score
            - latency_ms: LLM inference time
            - risk_score: cumulative risk score (if call_id provided)
            - risk_status: cumulative risk status (if call_id provided)
        """
        # 1. Sliding Window (Last 2000 chars) for the LLM.
        # Using a larger window reduces the chance we cut away the key scam cue.
        llm_text = text[-2000:] if len(text) > 2000 else text

        # Calibrated probability (JSON) generally separates normal vs vishing better than
        # a forced YES/NO prompt (which can be biased).
        prompt = prompts.build_vishing_probability_prompt(llm_text)
        prob_yes, latency_ms = await self.client.get_vishing_probability(prompt)
        
        # Heuristics are better on the full text to catch early signs
        heuristic_score = heuristics.compute_heuristic_score(text)
        combined = heuristics.combine_scores(prob_yes, heuristic_score)

        result = {
            "score": combined,
            "prob_yes": prob_yes,
            "heuristic_score": heuristic_score,
            "latency_ms": latency_ms,
        }
        
        # 2. State Management (Risk Accumulator)
        if call_id:
            risk_score, risk_status = self.state_manager.update(call_id, prob_yes, heuristic_score)
            result["risk_score"] = risk_score
            result["risk_status"] = risk_status
            
        return result

    async def classify_batch(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Classify a batch of texts and compute aggregate metrics.

        Args:
            texts: list of transcript texts
            labels: optional ground-truth labels (0/1) for metric computation

        Returns dict with:
            - results: per-text classification results
            - metrics: aggregate metrics (if labels provided)
        """
        import asyncio

        tasks = [self.classify(text) for text in texts]
        results = await asyncio.gather(*tasks)

        scores = [r["score"] for r in results]
        scores = heuristics.min_max_normalize(scores)

        # Update normalized scores in results
        for i, r in enumerate(results):
            r["score"] = scores[i]

        output: Dict[str, Any] = {"results": list(results)}

        if labels is not None and len(labels) == len(texts):
            threshold_info = find_best_thresholds(labels, scores)
            roc_auc = compute_roc_auc(labels, scores)
            output["metrics"] = {
                "best_f1": threshold_info["best_f1"],
                "recall_priority": threshold_info["recall_priority"],
                "roc_auc": roc_auc,
            }

        return output


# --- Metric evaluation functions (ported from experiment1_mistral.py) ---


def evaluate_at_threshold(
    labels: List[int],
    scores: List[float],
    threshold: float,
) -> Dict[str, Any]:
    """
    Compute confusion matrix and basic metrics at a given threshold.
    """
    y_true = labels
    y_pred = [1 if s >= threshold else 0 for s in scores]

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def find_best_thresholds(
    labels: List[int],
    scores: List[float],
) -> Dict[str, Any]:
    """
    Search thresholds in [0.05, 0.95] (step 0.01) to find:

    - best_f1_threshold: maximizes F1-score
    - recall_priority_threshold: maximizes Recall with Precision >= 0.5
    """
    if not labels:
        return {"best_f1": None, "recall_priority": None}

    thresholds = [round(t, 2) for t in [0.05 + 0.01 * i for i in range(91)]]

    best_f1_metric: Dict[str, Any] = {}
    best_f1_value = -1.0

    best_recall_metric: Dict[str, Any] = {}
    best_recall_value = -1.0

    for t in thresholds:
        m = evaluate_at_threshold(labels, scores, t)
        f1_val = m["f1"]
        rec_val = m["recall"]
        prec_val = m["precision"]

        if f1_val > best_f1_value:
            best_f1_value = f1_val
            best_f1_metric = m

        if prec_val >= 0.5 and rec_val > best_recall_value:
            best_recall_value = rec_val
            best_recall_metric = m

    if not best_recall_metric:
        best_recall_metric = best_f1_metric

    return {
        "best_f1": best_f1_metric,
        "recall_priority": best_recall_metric,
    }


def compute_roc_auc(labels: List[int], scores: List[float]) -> float:
    """Compute ROC-AUC. Returns NaN if undefined."""
    if not labels:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")
