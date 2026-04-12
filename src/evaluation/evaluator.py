"""
evaluation/evaluator.py
Computes both hallucination metrics and standard QA metrics (F1, EM).
Supports baseline vs fine-tuned comparison.
"""

from __future__ import annotations

import json
import re
import string
from datetime import datetime, timezone
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional
import csv

import numpy as np
from loguru import logger

from ..diagnosis.metric_engine import HallucinationMetrics


# ── QA metric helpers (HotpotQA / NQ style) ──────────────────────────────────
def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Result containers ─────────────────────────────────────────────────────────
@dataclass
class SampleResult:
    query: str
    prediction: str
    ground_truth: str
    f1: float
    em: float
    metrics: HallucinationMetrics


@dataclass
class AggregateResult:
    avg_f1: float
    avg_em: float
    avg_scr: float
    avg_cr: float
    avg_tve: float
    avg_cdee: float
    avg_chs: float
    n_samples: int
    model_tag: str = "baseline"
    avg_retrieval_conflict: float = 0.0
    avg_overgeneralization: float = 0.0
    avg_outdated_information: float = 0.0
    avg_synthesis_error: float = 0.0
    avg_overall_hallucination_score: float = 0.0
    avg_confidence: float = 0.0
    pct_hallucinated_outputs: float = 0.0

    def summary(self) -> str:
        return (
            f"[{self.model_tag}] N={self.n_samples} | "
            f"F1={self.avg_f1:.3f} | EM={self.avg_em:.3f} | "
            f"SCR={self.avg_scr:.3f} | CR={self.avg_cr:.3f} | "
            f"TVE={self.avg_tve:.3f} | CDEE={self.avg_cdee:.3f} | "
            f"CHS={self.avg_chs:.3f} | OHS={self.avg_overall_hallucination_score:.3f} | "
            f"H%={self.pct_hallucinated_outputs:.1f}"
        )


# ── Evaluator ─────────────────────────────────────────────────────────────────
class RAGEvaluator:
    """
    Evaluates a list of (query, prediction, ground_truth, hallucination_metrics)
    and aggregates into AggregateResult.
    """

    def __init__(self, results_dir: str = "outputs/eval"):
        self.results_dir = results_dir
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        queries: List[str],
        predictions: List[str],
        ground_truths: List[str],
        hallucination_metrics: List[HallucinationMetrics],
        model_tag: str = "model",
    ) -> AggregateResult:
        assert len(queries) == len(predictions) == len(ground_truths) == len(hallucination_metrics)

        sample_results = []
        for q, pred, gt, hm in zip(queries, predictions, ground_truths, hallucination_metrics):
            sr = SampleResult(
                query=q,
                prediction=pred,
                ground_truth=gt,
                f1=f1_score(pred, gt),
                em=exact_match(pred, gt),
                metrics=hm,
            )
            sample_results.append(sr)

        n = len(sample_results)
        agg = AggregateResult(
            avg_f1=np.mean([r.f1 for r in sample_results]),
            avg_em=np.mean([r.em for r in sample_results]),
            avg_scr=np.mean([r.metrics.scr for r in sample_results]),
            avg_cr=np.mean([r.metrics.cr for r in sample_results]),
            avg_tve=np.mean([r.metrics.tve for r in sample_results]),
            avg_cdee=np.mean([r.metrics.cdee for r in sample_results]),
            avg_chs=np.mean([r.metrics.chs for r in sample_results]),
            n_samples=n,
            model_tag=model_tag,
            avg_retrieval_conflict=np.mean([r.metrics.retrieval_conflict for r in sample_results]),
            avg_overgeneralization=np.mean([r.metrics.overgeneralization for r in sample_results]),
            avg_outdated_information=np.mean([r.metrics.outdated_information for r in sample_results]),
            avg_synthesis_error=np.mean([r.metrics.synthesis_error for r in sample_results]),
            avg_overall_hallucination_score=np.mean(
                [r.metrics.overall_hallucination_score for r in sample_results]
            ),
            avg_confidence=np.mean([r.metrics.confidence for r in sample_results]),
            pct_hallucinated_outputs=100.0
            * np.mean(
                [
                    r.metrics.overall_hallucination_score >= 0.5
                    or r.metrics.retrieval_conflict_label
                    or r.metrics.overgeneralization_label
                    or r.metrics.outdated_information_label
                    or r.metrics.synthesis_error_label
                    for r in sample_results
                ]
            ),
        )
        logger.info(agg.summary())
        self._save(sample_results, agg, model_tag)
        return agg

    def compare(self, baseline: AggregateResult, tuned: AggregateResult) -> dict:
        """Print Δ metrics between baseline and fine-tuned model."""
        delta = {
            "Δ F1":   round(tuned.avg_f1 - baseline.avg_f1, 4),
            "Δ EM":   round(tuned.avg_em - baseline.avg_em, 4),
            "Δ SCR":  round(tuned.avg_scr - baseline.avg_scr, 4),
            "Δ CR":   round(tuned.avg_cr - baseline.avg_cr, 4),
            "Δ TVE":  round(tuned.avg_tve - baseline.avg_tve, 4),
            "Δ CDEE": round(tuned.avg_cdee - baseline.avg_cdee, 4),
            "Δ CHS":  round(tuned.avg_chs - baseline.avg_chs, 4),
            "Δ OHS": round(
                tuned.avg_overall_hallucination_score - baseline.avg_overall_hallucination_score,
                4,
            ),
        }
        logger.info("=== Comparison ===")
        for k, v in delta.items():
            sign = "▲" if v > 0 else ("▼" if v < 0 else "—")
            logger.info(f"  {k}: {sign} {abs(v):.4f}")
        return delta

    def _save(self, sample_results: List[SampleResult], agg: AggregateResult, tag: str) -> None:
        # Per-sample results
        sample_path = Path(self.results_dir) / f"{tag}_samples.jsonl"
        with open(sample_path, "w") as f:
            for r in sample_results:
                row = {
                    "query": r.query,
                    "prediction": r.prediction,
                    "ground_truth": r.ground_truth,
                    "f1": r.f1,
                    "em": r.em,
                    **asdict(r.metrics),
                }
                f.write(json.dumps(row) + "\n")

        # Aggregate
        agg_path = Path(self.results_dir) / f"{tag}_aggregate.json"
        with open(agg_path, "w") as f:
            json.dump(asdict(agg), f, indent=2)

        self._append_hallucination_logs(sample_results=sample_results, model_tag=tag)

        logger.info(f"Results saved to {self.results_dir}")

    def _append_hallucination_logs(self, sample_results: List[SampleResult], model_tag: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()

        jsonl_path = Path(self.results_dir) / "hallucination_samples.jsonl"
        with open(jsonl_path, "a") as f:
            for r in sample_results:
                row = {
                    "query": r.query,
                    "response": r.prediction,
                    "scores": r.metrics.hallucination_scores(),
                    "labels": r.metrics.hallucination_labels(),
                    "confidence": r.metrics.confidence,
                    "overall_hallucination_score": r.metrics.overall_hallucination_score,
                    "timestamp": ts,
                    "model_tag": model_tag,
                }
                f.write(json.dumps(row) + "\n")

        csv_path = Path(self.results_dir) / "hallucination_samples.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "model_tag",
                    "query",
                    "overall_score",
                    "retrieval_conflict",
                    "overgeneralization",
                    "outdated_information",
                    "synthesis_error",
                    "retrieval_conflict_label",
                    "overgeneralization_label",
                    "outdated_information_label",
                    "synthesis_error_label",
                    "confidence",
                ],
            )
            if write_header:
                writer.writeheader()
            for r in sample_results:
                writer.writerow(
                    {
                        "timestamp": ts,
                        "model_tag": model_tag,
                        "query": r.query,
                        "overall_score": r.metrics.overall_hallucination_score,
                        "retrieval_conflict": r.metrics.retrieval_conflict,
                        "overgeneralization": r.metrics.overgeneralization,
                        "outdated_information": r.metrics.outdated_information,
                        "synthesis_error": r.metrics.synthesis_error,
                        "retrieval_conflict_label": r.metrics.retrieval_conflict_label,
                        "overgeneralization_label": r.metrics.overgeneralization_label,
                        "outdated_information_label": r.metrics.outdated_information_label,
                        "synthesis_error_label": r.metrics.synthesis_error_label,
                        "confidence": r.metrics.confidence,
                    }
                )
