"""
retrieval/retriever.py
High-level Retriever that combines DenseEncoder + FAISSIndex.
"""

from __future__ import annotations

import json
from typing import List, Optional

from loguru import logger

from .encoder import DenseEncoder, FAISSIndex, build_index_from_corpus


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

        def _corpus_count(path: str) -> int:
            count = 0
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        json.loads(line)
                        count += 1
            return count

        def _load_or_rebuild_index(encoder: DenseEncoder):
            if os.path.exists(index_path):
                idx = FAISSIndex.load(index_path)
                return idx

            if allow_rebuild_if_missing:
                logger.warning(
                    f"FAISS index missing at {index_path}; rebuilding from corpus {corpus_path}"
                )
                return build_index_from_corpus(
                    corpus_path=corpus_path,
                    encoder=encoder,
                    index_path=index_path,
                    batch_size=cfg.get("index_build_batch_size", 128),
                )

            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. Run scripts/build_index.py first."
            )

        enc = DenseEncoder(
            model_name=cfg["encoder_model"],
            cache_dir=cfg.get("cache_dir"),
        )
        index_path = cfg["index_path"]
        corpus_path = cfg["corpus_path"]
        allow_rebuild_if_missing = bool(cfg.get("allow_rebuild_if_missing", True))
        allow_rebuild_if_stale = bool(cfg.get("allow_rebuild_if_stale", True))

        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"Corpus not found at {corpus_path}. Retrieval grounding cannot proceed."
            )

        index = _load_or_rebuild_index(enc)

        expected_docs = _corpus_count(corpus_path)
        indexed_docs = len(index.doc_ids)
        if expected_docs != indexed_docs:
            msg = (
                f"Index/corpus mismatch: corpus has {expected_docs} docs but index has {indexed_docs}."
            )
            if allow_rebuild_if_stale:
                logger.warning(msg + " Rebuilding index from corpus.")
                index = build_index_from_corpus(
                    corpus_path=corpus_path,
                    encoder=enc,
                    index_path=index_path,
                    batch_size=cfg.get("index_build_batch_size", 128),
                )
            else:
                raise ValueError(msg)

        return cls(encoder=enc, index=index, top_k=cfg.get("top_k", 5))