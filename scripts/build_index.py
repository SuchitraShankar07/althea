#!/usr/bin/env python
"""
scripts/build_index.py

Download a Wikipedia subset (or use a local JSONL corpus)
and build the FAISS retrieval index.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from loguru import logger
from src.configuration import load_config_file

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.encoder import DenseEncoder, build_index_from_corpus


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_wikipedia_sample(output_path: str, n_docs: int = 50_000) -> None:
    """
    Download Wikipedia sample OR fallback to dummy corpus.
    """
    try:
        from datasets import load_dataset

        logger.info(f"Downloading Wikipedia sample ({n_docs:,} docs)...")

        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for i, row in enumerate(ds):
                if i >= n_docs:
                    break

                doc = {
                    "id": row.get("id", str(i)),
                    "title": row.get("title", f"Article_{i}"),
                    "text": row.get("text", "")[:1500],
                }

                f.write(json.dumps(doc) + "\n")

                if i % 1000 == 0 and i > 0:
                    logger.info(f"Downloaded {i:,} documents...")

        logger.info(f"Saved {n_docs:,} docs to {output_path}")

    except Exception as e:
        logger.warning(f"Wikipedia download failed: {e}")
        logger.info("Falling back to dummy corpus...")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for i in range(min(n_docs, 1000)):
                doc = {
                    "id": f"doc_{i}",
                    "title": f"Sample Document {i}",
                    "text": (
                        f"This is sample document {i} with synthetic content "
                        f"for retrieval testing. " * 5
                    ),
                }
                f.write(json.dumps(doc) + "\n")

        logger.info("Dummy corpus created successfully")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--n_docs", type=int, default=10_000)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    # Device handling
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    try:
        cfg = load_config_file(args.config)
    except FileNotFoundError:
        logger.warning("Config not found → using fallback config")

        cfg = {
            "retrieval": {
                "corpus_path": "data/wiki.jsonl",
                "index_path": "data/faiss_index",
                "encoder_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "paths": {"cache_dir": "cache"},
        }

    ret_cfg = cfg["retrieval"]

    corpus_path = args.corpus or ret_cfg["corpus_path"]
    index_path = ret_cfg["index_path"]

    # ========================================================
    # PREPARE DATA
    # ========================================================

    if not Path(corpus_path).exists():
        logger.info("Corpus not found → downloading...")
        download_wikipedia_sample(corpus_path, args.n_docs)

    # ========================================================
    # BUILD INDEX
    # ========================================================

    try:
        encoder = DenseEncoder(
            model_name=ret_cfg["encoder_model"],
            device=device,
            cache_dir=cfg.get("paths", {}).get("cache_dir"),
        )

        batch_size = 512 if device == "cuda" else 128

        logger.info("Building FAISS index...")

        build_index_from_corpus(
            corpus_path=corpus_path,
            encoder=encoder,
            index_path=index_path,
            batch_size=batch_size,
        )

        logger.success("✓ FAISS index built successfully")

    except Exception as e:
        logger.error(f"Index build failed: {e}")
        logger.error("Check retrieval module implementation.")


# ============================================================

if __name__ == "__main__":
    main()