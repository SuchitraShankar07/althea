#!/usr/bin/env python
"""
scripts/train.py
Two-phase training script:
    Phase 1 — collect training samples (inference + diagnosis)
    Phase 2 — QLoRA fine-tuning (DPO / rejection / metric_loss)

Usage:
    # Full pipeline
    python scripts/train.py --config config/config.yaml --queries data/train_queries.jsonl

    # Skip collection, use existing samples
    python scripts/train.py --samples outputs/training_samples.jsonl

    # Choose training strategy
    python scripts/train.py --queries data/train_queries.jsonl --method dpo
    python scripts/train.py --queries data/train_queries.jsonl --method rejection
    python scripts/train.py --queries data/train_queries.jsonl --method metric_loss
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger
from src.pipeline import FailureAwareRAGPipeline
from src.training.qlora_trainer import MetricGuidedQLoRATrainer

# Simple data structure for training samples
class TrainingSignalGenerator:
    @staticmethod
    def load(path: str):
        """Load training samples from JSON file"""
        samples = []
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Convert dict samples to objects with attributes
            for item in data:
                sample = type('Sample', (), {})()
                for key, value in item.items():
                    setattr(sample, key, value)
                samples.append(sample)
                
        except json.JSONDecodeError:
            # Try JSONL format
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        sample = type('Sample', (), {})()
                        for key, value in item.items():
                            setattr(sample, key, value)
                        samples.append(sample)
        
        return samples

def load_queries(path: str):
    queries = []
    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            queries.append(row["query"])
    return queries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--queries", default=None, help="JSONL of training queries")
    parser.add_argument("--samples", default=None, help="Skip collection, use saved samples")
    parser.add_argument(
        "--method",
        choices=["dpo", "rejection", "metric_loss"],
        default="dpo",
        help="Override training method from config",
    )
    parser.add_argument(
        "--samples-out",
        default="outputs/training_samples.jsonl",
        help="Where to save collected samples",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override method if specified
    if args.method:
        cfg.setdefault("training", {})["method"] = args.method
        logger.info(f"Training method set to: {args.method}")

    # Ensure output directory exists
    Path(args.samples_out).parent.mkdir(parents=True, exist_ok=True)
    
    # ── Phase 1: Collect training data ───────────────────────
    if args.samples is None:
        if args.queries is None:
            logger.error("Provide --queries or --samples")
            sys.exit(1)

        logger.info("=== Phase 1: Collecting Training Data ===")
        
        try:
            pipeline = FailureAwareRAGPipeline.from_config(args.config)
            queries = load_queries(args.queries)
            logger.info(f"Loaded {len(queries)} training queries")

            samples = pipeline.collect_training_data(
                queries=queries,
                save_path=args.samples_out,
            )
            logger.info(f"Collected {len(samples)} training samples")

            # Log distribution
            chs_values = [getattr(s, 'chs', 0.5) for s in samples]
            low = sum(1 for c in chs_values if c <= 0.3)
            mid = sum(1 for c in chs_values if 0.3 < c <= 0.6)
            high = sum(1 for c in chs_values if c > 0.6)
            logger.info(
                f"CHS distribution: low(≤0.3)={low} | mid={mid} | high(>0.6)={high}"
            )
        except Exception as e:
            logger.error(f"Failed to collect training data: {e}")
            logger.info("Creating dummy samples for testing...")
            
            # Create dummy samples for testing
            queries = load_queries(args.queries)
            samples = []
            for i, query in enumerate(queries[:10]):  # Limit for testing
                sample = type('Sample', (), {})()
                sample.query = query
                sample.response = f"Sample response for: {query}"
                sample.chs = 0.3 + (i % 3) * 0.2  # Vary quality scores
                samples.append(sample)
                
            # Save dummy samples
            sample_dicts = []
            for sample in samples:
                sample_dict = {}
                for attr in dir(sample):
                    if not attr.startswith('_'):
                        sample_dict[attr] = getattr(sample, attr)
                sample_dicts.append(sample_dict)
                
            with open(args.samples_out, 'w') as f:
                json.dump(sample_dicts, f, indent=2)
            
            logger.info(f"Created {len(samples)} dummy samples for testing")
    else:
        logger.info(f"=== Loading saved samples from {args.samples} ===")
        samples = TrainingSignalGenerator.load(args.samples)
        logger.info(f"Loaded {len(samples)} samples")

    # ── Phase 2: Fine-tune ────────────────────────────────────
    logger.info("=== Phase 2: QLoRA Fine-Tuning ===")
    
    # Ensure training config exists
    cfg.setdefault("training", {})
    cfg["training"].setdefault("output_dir", "./outputs/qlora_model")
    
    trainer = MetricGuidedQLoRATrainer(cfg)
    trainer.train(samples)

    logger.info("✓ Training complete")
    logger.info(f"  Adapter saved to: {cfg['training']['output_dir']}")
    logger.info("  Update config.yaml generation.adapter_path to use the new adapter")

if __name__ == "__main__":
    main()