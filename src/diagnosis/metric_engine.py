"""Metric engine: aggregates per-claim verdicts into diagnostic scores."""

from typing import Dict, List


class MetricEngine:
    """Computes hallucination and faithfulness metrics over a set of claim verdicts."""

    def compute(self, verdicts: List[Dict]) -> Dict[str, float]:
        """Aggregate claim-level verification results into response-level scores.

        Args:
            verdicts: A list of dicts as returned by
                :meth:`~src.diagnosis.VerificationEngine.verify`.

        Returns:
            A dict containing:
            - ``faithfulness``: fraction of supported claims.
            - ``hallucination_rate``: fraction of contradicted claims.
            - ``coverage``: fraction of claims with sufficient evidence.
        """
        if not verdicts:
            return {"faithfulness": 0.0, "hallucination_rate": 0.0, "coverage": 0.0}

        n = len(verdicts)
        supported = sum(1 for v in verdicts if v["verdict"] == "supported")
        contradicted = sum(1 for v in verdicts if v["verdict"] == "contradicted")
        covered = sum(1 for v in verdicts if v["verdict"] != "insufficient")

        return {
            "faithfulness": supported / n,
            "hallucination_rate": contradicted / n,
            "coverage": covered / n,
        }
