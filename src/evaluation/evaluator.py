"""Evaluator module: end-to-end RAG quality evaluation."""

from typing import Dict, List, Optional


class Evaluator:
    """Evaluates a RAG pipeline's outputs across multiple quality dimensions."""

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Args:
            metrics: Which metrics to compute. Supported values:
                ``"faithfulness"``, ``"answer_relevance"``, ``"context_recall"``.
                Defaults to all three.
        """
        self.metrics = metrics or ["faithfulness", "answer_relevance", "context_recall"]

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate a single query-answer pair.

        Args:
            query: The original user question.
            answer: The generated answer to evaluate.
            contexts: The retrieved passages used to produce *answer*.
            ground_truth: An optional reference answer for supervised metrics.

        Returns:
            A dict mapping each requested metric name to a float score in ``[0, 1]``.
        """
        scores: Dict[str, float] = {}

        if "faithfulness" in self.metrics:
            scores["faithfulness"] = self._faithfulness(answer, contexts)

        if "answer_relevance" in self.metrics:
            scores["answer_relevance"] = self._answer_relevance(query, answer)

        if "context_recall" in self.metrics:
            scores["context_recall"] = self._context_recall(
                contexts, ground_truth or ""
            )

        return scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Compute a simple word-overlap proxy for faithfulness."""
        if not contexts:
            return 0.0
        context_tokens = set(" ".join(contexts).lower().split())
        answer_tokens = answer.lower().split()
        if not answer_tokens:
            return 0.0
        overlap = sum(1 for t in answer_tokens if t in context_tokens)
        return overlap / len(answer_tokens)

    def _answer_relevance(self, query: str, answer: str) -> float:
        """Compute a simple word-overlap proxy for answer relevance."""
        query_tokens = set(query.lower().split())
        answer_tokens = answer.lower().split()
        if not query_tokens or not answer_tokens:
            return 0.0
        overlap = sum(1 for t in answer_tokens if t in query_tokens)
        return overlap / len(query_tokens)

    def _context_recall(self, contexts: List[str], ground_truth: str) -> float:
        """Compute a simple word-overlap proxy for context recall."""
        if not ground_truth or not contexts:
            return 0.0
        gt_tokens = set(ground_truth.lower().split())
        context_tokens = set(" ".join(contexts).lower().split())
        if not gt_tokens:
            return 0.0
        overlap = len(gt_tokens & context_tokens)
        return overlap / len(gt_tokens)
