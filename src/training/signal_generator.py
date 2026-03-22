"""Signal generator: produces training signals from diagnostic outputs."""

from typing import Dict, List


class SignalGenerator:
    """Converts diagnosis results into preference pairs for RLHF / DPO fine-tuning."""

    def __init__(self, faithfulness_threshold: float = 0.7):
        """
        Args:
            faithfulness_threshold: Minimum faithfulness score for a response to be
                considered a *chosen* sample.
        """
        self.faithfulness_threshold = faithfulness_threshold

    def generate_signals(
        self,
        query: str,
        response: str,
        diagnosis: Dict,
        alternative_response: str = "",
    ) -> Dict:
        """Produce a training signal dict for a single query-response pair.

        Args:
            query: The original user question.
            response: The model-generated answer.
            diagnosis: Diagnostic metrics dict (e.g. from :class:`MetricEngine`).
            alternative_response: An optional lower-quality response to act as the
                *rejected* side of a preference pair.

        Returns:
            A dict with keys ``"prompt"``, ``"chosen"``, ``"rejected"``, and
            ``"scores"``.
        """
        faithfulness = diagnosis.get("faithfulness", 0.0)
        is_chosen = faithfulness >= self.faithfulness_threshold

        if not alternative_response:
            # Cannot form a valid preference pair without a contrastive response.
            return {
                "prompt": query,
                "chosen": response,
                "rejected": None,
                "scores": diagnosis,
            }

        return {
            "prompt": query,
            "chosen": response if is_chosen else alternative_response,
            "rejected": alternative_response if is_chosen else response,
            "scores": diagnosis,
        }

    def batch_generate(
        self,
        samples: List[Dict],
    ) -> List[Dict]:
        """Generate signals for a batch of samples.

        Args:
            samples: A list of dicts, each containing ``"query"``, ``"response"``,
                ``"diagnosis"``, and optionally ``"alternative_response"`` keys.

        Returns:
            A list of signal dicts.
        """
        return [
            self.generate_signals(
                query=s["query"],
                response=s["response"],
                diagnosis=s["diagnosis"],
                alternative_response=s.get("alternative_response", ""),
            )
            for s in samples
        ]
