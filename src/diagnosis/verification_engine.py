"""Verification engine: assesses whether claims are supported by evidence."""

from typing import Dict, List


class VerificationEngine:
    """Verifies each claim against retrieved evidence using an NLI model."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"

    def __init__(
        self,
        entailment_model: str = "cross-encoder/nli-deberta-v3-base",
        confidence_threshold: float = 0.5,
    ):
        self.entailment_model = entailment_model
        self.confidence_threshold = confidence_threshold
        self._model = None

    def _load_model(self):
        from transformers import pipeline

        self._model = pipeline("text-classification", model=self.entailment_model)

    def verify(
        self, claim: str, evidence_passages: List[Dict]
    ) -> Dict:
        """Assess whether *claim* is supported, contradicted, or inconclusive.

        Args:
            claim: A single atomic claim.
            evidence_passages: A list of document dicts (must contain a ``"text"`` key).

        Returns:
            A dict with keys ``"verdict"``, ``"score"``, and ``"evidence"``.
        """
        if not evidence_passages:
            return {"verdict": self.INSUFFICIENT, "score": 0.0, "evidence": []}

        if self._model is None:
            self._load_model()

        best_label, best_score = self.INSUFFICIENT, 0.0
        for doc in evidence_passages:
            passage = doc.get("text", "")
            if not passage:
                continue
            result = self._model(f"{passage} [SEP] {claim}")[0]
            label = result["label"].lower()
            score = result["score"]
            if label == "entailment" and score > best_score:
                best_label, best_score = self.SUPPORTED, score
            elif label == "contradiction" and score > best_score:
                best_label, best_score = self.CONTRADICTED, score

        return {
            "verdict": best_label,
            "score": best_score,
            "evidence": [d.get("text", "") for d in evidence_passages],
        }
