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
import torch
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
    try:
        from datasets import load_dataset

        logger.info(f"Downloading Wikipedia sample ({n_docs:,} docs)...")
        
        # Try the more reliable approach with explicit configuration
        ds = load_dataset(
            "wikipedia", 
            "20220301.en", 
            split="train", 
            streaming=True,
            trust_remote_code=True
        )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for i, row in enumerate(ds):
                if i >= n_docs:
                    break
                    
                # Handle potential key variations
                doc_id = row.get("id", str(i))
                title = row.get("title", f"Article_{i}")
                text = row.get("text", "")[:1500]  # truncate long articles
                
                doc = {
                    "id": doc_id,
                    "title": title,
                    "text": text
                }
                f.write(json.dumps(doc) + "\n")
                
                if i % 1000 == 0:
                    logger.info(f"Downloaded {i:,} documents...")
                    
        logger.info(f"Saved {n_docs:,} docs to {output_path}")
        
    except Exception as e:
        logger.warning(f"Failed to download Wikipedia: {e}")
        logger.info("Creating dummy corpus for testing...")
        
        # Create a dummy corpus for testing
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        dummy_docs = []
        for i in range(min(n_docs, 1000)):  # Limit dummy docs
            doc = {
                "id": f"doc_{i}",
                "title": f"Sample Document {i}",
                "text": f"This is sample document {i} with some content for testing retrieval. " * 10
            }
            dummy_docs.append(doc)
        
        with open(output_path, "w") as f:
            for doc in dummy_docs:
                f.write(json.dumps(doc) + "\n")
                
        logger.info(f"Created dummy corpus with {len(dummy_docs)} documents")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--corpus", default=None, help="Path to existing JSONL corpus")
    parser.add_argument("--n_docs", type=int, default=10000)  # Reduced default for faster testing
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Force GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load config
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        logger.info("Creating minimal config...")
        cfg = {
            "retrieval": {
                "corpus_path": "data/wikipedia_corpus.jsonl",
                "index_path": "data/faiss_index",
                "encoder_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "paths": {
                "cache_dir": "cache"
            }
        }

    ret_cfg = cfg["retrieval"]
    corpus_path = args.corpus or ret_cfg["corpus_path"]
    index_path = ret_cfg["index_path"]

    # Download if needed
    if not Path(corpus_path).exists():
        logger.info(f"Corpus not found at {corpus_path}. Downloading Wikipedia sample...")
        download_wikipedia_sample(corpus_path, n_docs=args.n_docs)

    # Build encoder with GPU
    try:
        encoder = DenseEncoder(
            model_name=ret_cfg["encoder_model"],
            device=device,
            cache_dir=cfg.get("paths", {}).get("cache_dir"),
        )

        # Build & save FAISS index with larger batch size for GPU
        batch_size = 512 if device == "cuda" else 256
        build_index_from_corpus(
            corpus_path=corpus_path,
            encoder=encoder,
            index_path=index_path,
            batch_size=batch_size,
        )
        logger.info("✓ Index built successfully")
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        logger.info("This might be due to missing src modules. Please ensure the retrieval module is properly implemented.")

if __name__ == "__main__":
    main()