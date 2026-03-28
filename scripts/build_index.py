#!/usr/bin/env python
"""
scripts/build_index.py
Download a Wikipedia subset (or use a local JSONL corpus) and
build the FAISS retrieval index.

Usage:
    python scripts/build_index.py --config config/config.yaml
    python scripts/build_index.py --corpus data/my_corpus.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger

from src.retrieval.encoder import DenseEncoder, build_index_from_corpus


def download_wikipedia_sample(output_path: str, n_docs: int = 50_000) -> None:
    """
    Download a small Wikipedia sample from HuggingFace datasets
    and save as JSONL.
    """
    from datasets import load_dataset

    logger.info(f"Downloading Wikipedia sample ({n_docs:,} docs)...")
    ds = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, row in enumerate(ds):
            if i >= n_docs:
                break
            doc = {
                "id": row["id"],
                "title": row["title"],
                "text": row["text"][:1500],   # truncate long articles
            }
            f.write(json.dumps(doc) + "\n")
    logger.info(f"Saved {n_docs:,} docs to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--corpus", default=None, help="Path to existing JSONL corpus")
    parser.add_argument("--n_docs", type=int, default=50_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ret_cfg = cfg["retrieval"]
    corpus_path = args.corpus or ret_cfg["corpus_path"]
    index_path = ret_cfg["index_path"]

    # Download if needed
    if not Path(corpus_path).exists():
        logger.info(f"Corpus not found at {corpus_path}. Downloading Wikipedia sample...")
        download_wikipedia_sample(corpus_path, n_docs=args.n_docs)

    # Build encoder
    encoder = DenseEncoder(
        model_name=ret_cfg["encoder_model"],
        device=args.device,
        cache_dir=cfg.get("paths", {}).get("cache_dir"),
    )

    # Build & save FAISS index
    build_index_from_corpus(
        corpus_path=corpus_path,
        encoder=encoder,
        index_path=index_path,
        batch_size=256,
    )
    logger.info("✓ Index built successfully")


if __name__ == "__main__":
    main()
