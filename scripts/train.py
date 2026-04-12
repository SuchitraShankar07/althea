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

from loguru import logger
from src.pipeline import FailureAwareRAGPipeline
from src.configuration import load_config_file
from src.training.qlora_trainer import MetricGuidedQLoRATrainer
from src.training.signal_generator import TrainingSignalGenerator

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
        "--skip-collection",
        action="store_true",
        help=(
            "Skip Phase 1 data collection and proceed directly to fine-tuning. "
            "Uses --samples if provided, otherwise --samples-out."
        ),
    )
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
    parser.add_argument(
        "--disable-hallucination-eval",
        action="store_true",
        help="Disable extended hallucination taxonomy evaluation while collecting training data.",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config_file(args.config)

    # Override method if specified
    if args.method:
        cfg.setdefault("training", {})["method"] = args.method
        logger.info(f"Training method set to: {args.method}")

    # Ensure output directory exists
    Path(args.samples_out).parent.mkdir(parents=True, exist_ok=True)
    
    sample_path = args.samples
    if args.skip_collection:
        sample_path = sample_path or args.samples_out
        if not Path(sample_path).exists():
            logger.error(
                "--skip-collection was set but no samples file was found at: "
                f"{sample_path}"
            )
            logger.error(
                "Run once without --skip-collection to generate samples, "
                "or pass --samples <path>."
            )
            sys.exit(1)

    # ── Phase 1: Collect training data ───────────────────────
    if sample_path is None:
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
                enable_hallucination_eval=not args.disable_hallucination_eval,
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
            logger.error("Training data collection failed; refusing to create dummy samples.")
            sys.exit(1)
    else:
        logger.info(f"=== Loading saved samples from {sample_path} ===")
        samples = TrainingSignalGenerator.load(sample_path)
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