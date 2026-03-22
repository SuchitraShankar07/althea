"""Generator module for LLM-based answer generation."""

from typing import List, Optional


class Generator:
    """Generates answers conditioned on retrieved context using a causal LLM."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._pipeline = None

    def _load_pipeline(self):
        """Lazily load the HuggingFace text-generation pipeline."""
        from transformers import pipeline

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def _build_prompt(self, query: str, context_passages: List[str]) -> str:
        context = "\n\n".join(context_passages)
        return (
            f"[INST] Answer the following question using only the provided context.\n\n"
            f"Context:\n{context}\n\nQuestion: {query} [/INST]"
        )

    def generate(
        self,
        query: str,
        context_passages: Optional[List[str]] = None,
    ) -> str:
        """Generate an answer for *query* grounded in *context_passages*.

        Args:
            query: The user question.
            context_passages: A list of relevant text passages retrieved for the query.

        Returns:
            The generated answer string.
        """
        if self._pipeline is None:
            self._load_pipeline()

        prompt = self._build_prompt(query, context_passages or [])
        outputs = self._pipeline(prompt)
        generated = outputs[0]["generated_text"]
        # Strip the prompt prefix that some pipelines echo back.
        return generated.removeprefix(prompt).strip()
