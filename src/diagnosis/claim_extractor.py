"""Claim extractor module: decomposes generated text into atomic claims."""

from typing import List


class ClaimExtractor:
    """Extracts a list of atomic, verifiable claims from a piece of generated text."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name

    def extract(self, text: str) -> List[str]:
        """Break *text* into a list of atomic claims.

        Args:
            text: The model-generated answer to decompose.

        Returns:
            A list of claim strings, each expressing a single verifiable assertion.
        """
        # Split on sentence boundaries as a lightweight default; replace with an
        # LLM-based decomposition when a suitable model is available.
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]
