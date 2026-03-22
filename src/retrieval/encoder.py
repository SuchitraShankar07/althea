"""
retrieval/encoder.py
Encodes queries and documents into dense vectors using a
sentence-transformer model, then manages a FAISS index.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class DenseEncoder:
    """Wraps a SentenceTransformer for query / document encoding."""

    def __init__(self, model_name: str, device: str = "cpu", cache_dir: Optional[str] = None):
        logger.info(f"Loading encoder: {model_name}")
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
        self.device = device

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Return (N, D) float32 array."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


class FAISSIndex:
    """
    Flat inner-product FAISS index for approximate nearest-neighbour
    retrieval over a document corpus.
    """

    def __init__(self, embedding_dim: int):
        self.dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)   # cosine ~ IP on L2-norm vecs
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_metadata: List[dict] = []

    # ── Building ──────────────────────────────────────────────
    def add_documents(
        self,
        embeddings: np.ndarray,
        doc_ids: List[str],
        doc_texts: List[str],
        metadata: Optional[List[dict]] = None,
    ) -> None:
        assert embeddings.shape[0] == len(doc_ids) == len(doc_texts)
        self.index.add(embeddings)
        self.doc_ids.extend(doc_ids)
        self.doc_texts.extend(doc_texts)
        if metadata:
            self.doc_metadata.extend(metadata)
        else:
            self.doc_metadata.extend([{}] * len(doc_ids))
        logger.info(f"Index now contains {self.index.ntotal:,} vectors")

    # ── Querying ──────────────────────────────────────────────
    def search(
        self, query_emb: np.ndarray, top_k: int = 5
    ) -> List[dict]:
        """
        Returns list of dicts: {doc_id, text, score, metadata}
        query_emb: (1, D) or (D,) float32
        """
        if query_emb.ndim == 1:
            query_emb = query_emb[None, :]
        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "doc_id": self.doc_ids[idx],
                    "text": self.doc_texts[idx],
                    "score": float(score),
                    "metadata": self.doc_metadata[idx],
                }
            )
        return results

    # ── Persistence ───────────────────────────────────────────
    def save(self, index_path: str) -> None:
        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        meta_path = path.with_suffix(".pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "doc_ids": self.doc_ids,
                    "doc_texts": self.doc_texts,
                    "doc_metadata": self.doc_metadata,
                    "dim": self.dim,
                },
                f,
            )
        logger.info(f"Index saved to {path}")

    @classmethod
    def load(cls, index_path: str) -> "FAISSIndex":
        path = Path(index_path)
        meta_path = path.with_suffix(".pkl")
        faiss_index = faiss.read_index(str(path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        obj = cls(embedding_dim=meta["dim"])
        obj.index = faiss_index
        obj.doc_ids = meta["doc_ids"]
        obj.doc_texts = meta["doc_texts"]
        obj.doc_metadata = meta["doc_metadata"]
        logger.info(f"Loaded index with {faiss_index.ntotal:,} vectors from {path}")
        return obj


def build_index_from_corpus(
    corpus_path: str,
    encoder: DenseEncoder,
    index_path: str,
    batch_size: int = 256,
    text_field: str = "text",
    id_field: str = "id",
) -> FAISSIndex:
    """
    Read a JSONL corpus, encode every document, build and save the FAISS index.

    Corpus format (one JSON per line):
        {"id": "doc_1", "text": "...", "title": "...", ...}
    """
    logger.info(f"Building index from {corpus_path}")
    docs = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    if not docs:
        raise ValueError(f"Corpus at {corpus_path} is empty, cannot build FAISS index.")

    texts = [d[text_field] for d in docs]
    ids = [str(d.get(id_field, i)) for i, d in enumerate(docs)]
    meta = [{k: v for k, v in d.items() if k not in {text_field, id_field}} for d in docs]

    # Encode in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding corpus"):
        batch = texts[i : i + batch_size]
        all_embeddings.append(encoder.encode(batch, show_progress=False))
    embeddings = np.vstack(all_embeddings)

    # Infer dim from encoder output
    dim = embeddings.shape[1]
    index = FAISSIndex(embedding_dim=dim)
    index.add_documents(embeddings, ids, texts, meta)
    index.save(index_path)
    return index
