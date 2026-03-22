"""Evidence retriever module: fetches supporting passages for individual claims."""

from typing import Dict, List, Tuple


class EvidenceRetriever:
    """Retrieves evidence passages relevant to a given claim from a document store."""

    def __init__(self, retriever, top_k: int = 3):
        """
        Args:
            retriever: A :class:`~src.retrieval.Retriever` instance.
            top_k: Number of evidence passages to retrieve per claim.
        """
        self.retriever = retriever
        self.top_k = top_k

    def retrieve_evidence(
        self, claim: str
    ) -> List[Tuple[Dict, float]]:
        """Retrieve the most relevant passages for *claim*.

        Args:
            claim: A single atomic claim string.

        Returns:
            A list of ``(document_dict, score)`` tuples.
        """
        return self.retriever.retrieve(claim, top_k=self.top_k)
