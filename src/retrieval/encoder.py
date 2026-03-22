"""Encoder module for dense vector representations."""

from typing import List, Union

import numpy as np


class Encoder:
    """Encodes text into dense vector embeddings using a sentence-transformer model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazily load the underlying embedding model."""
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode one or more texts into embedding vectors.

        Args:
            texts: A single string or a list of strings to encode.
            batch_size: Number of texts to process per batch.
            show_progress_bar: Whether to display a progress bar.

        Returns:
            A numpy array of shape (n, embedding_dim).
        """
        if self._model is None:
            self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
