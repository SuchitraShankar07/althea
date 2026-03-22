"""Retriever module for nearest-neighbour document lookup."""

from typing import Dict, List, Tuple

import numpy as np


class Retriever:
    """Retrieves the most relevant documents for a given query using a FAISS index."""

    def __init__(self, encoder, index_path: str = "data/index", top_k: int = 5):
        """
        Args:
            encoder: An :class:`Encoder` instance used to embed queries.
            index_path: Path to a persisted FAISS index directory.
            top_k: Default number of documents to return per query.
        """
        self.encoder = encoder
        self.index_path = index_path
        self.top_k = top_k
        self._index = None
        self._documents: List[Dict] = []

    def load_index(self):
        """Load a FAISS index and associated document store from disk."""
        import faiss

        self._index = faiss.read_index(f"{self.index_path}/index.faiss")
        import json

        with open(f"{self.index_path}/documents.json", "r", encoding="utf-8") as fh:
            self._documents = json.load(fh)

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Dict, float]]:
        """Return the *top_k* most relevant documents for *query*.

        Args:
            query: The query string.
            top_k: Number of results to return; falls back to ``self.top_k``.

        Returns:
            A list of ``(document_dict, score)`` tuples ordered by relevance.
        """
        if self._index is None:
            self.load_index()

        k = top_k if top_k is not None else self.top_k
        query_vec = self.encoder.encode(query).astype(np.float32)
        distances, indices = self._index.search(query_vec.reshape(1, -1), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._documents[idx], float(dist)))
        return results
