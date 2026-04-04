"""
retrieval/retriever.py
High-level Retriever that combines DenseEncoder + FAISSIndex.
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from .encoder import DenseEncoder, FAISSIndex


class Retriever:
    """
    End-to-end retriever:
        query (str) → top-k document dicts

    Each returned dict:
        {doc_id, text, score, metadata}
    """

    def __init__(
        self,
        encoder: DenseEncoder,
        index: FAISSIndex,
        top_k: int = 5,
    ):
        self.encoder = encoder
        self.index = index
        self.top_k = top_k

    # ── Main API ──────────────────────────────────────────────
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        k = top_k or self.top_k
        q_emb = self.encoder.encode([query], normalize=True)[0]
        results = self.index.search(q_emb, top_k=k)
        return results

    def retrieve_batch(self, queries: List[str], top_k: Optional[int] = None) -> List[List[dict]]:
        k = top_k or self.top_k
        embeddings = self.encoder.encode(queries, normalize=True)
        return [self.index.search(emb, top_k=k) for emb in embeddings]

    # ── Factory ───────────────────────────────────────────────
    @classmethod
    def from_config(cls, cfg: dict) -> "Retriever":
        import os
        enc = DenseEncoder(
            model_name="phi3.5 by Microsoft",
            cache_dir=cfg.get("cache_dir"),
        )
        index_path = cfg["index_path"]
        if os.path.exists(index_path):
            index = FAISSIndex.load(index_path)
        else:
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run scripts/build_index.py first."
            )
        return cls(encoder=enc, index=index, top_k=cfg.get("top_k", 5))