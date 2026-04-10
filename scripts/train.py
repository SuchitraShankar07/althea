#!/usr/bin/env python
"""
scripts/train.py
Two-phase training script:
    Phase 1 — collect training samples (inference + diagnosis)
              Generates MULTIPLE answers per query for DPO pairing.
    Phase 2 — QLoRA fine-tuning (DPO / rejection / metric_loss)

Usage:
    # Full pipeline (collect data + train)
    python scripts/train.py --config config/config.yaml --queries data/train_queries.jsonl

    # Skip collection, use pre-collected samples
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
from src.training.signal_generator import TrainingSignalGenerator


def load_queries(path: str):
    """Load queries from JSONL file."""
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            queries.append(row["query"])
    return queries


def main():
    parser = argparse.ArgumentParser(
        description="Failure-Aware RAG training: collect data + QLoRA fine-tune"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--queries", default=None, help="JSONL of training queries")
    parser.add_argument(
        "--samples", default=None, help="Skip collection, use saved samples"
    )
    parser.add_argument(
        "--method",
        choices=["dpo", "rejection", "metric_loss"],
        default=None,
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

    # Override method if specified on CLI
    if args.method:
        cfg.setdefault("training", {})["method"] = args.method
        logger.info(f"Training method: {args.method}")

    # Ensure output directory exists
    Path(args.samples_out).parent.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Collect training data ───────────────────────
    if args.samples is None:
        if args.queries is None:
            logger.error("Provide --queries (JSONL file) or --samples (pre-collected)")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("Phase 1: Collecting Training Data")
        logger.info("=" * 60)

        pipeline = FailureAwareRAGPipeline.from_config(args.config)
        queries = load_queries(args.queries)
        logger.info(f"Loaded {len(queries)} training queries")

        samples = pipeline.collect_training_data(
            queries=queries,
            save_path=args.samples_out,
        )
        logger.info(f"Collected {len(samples)} training samples")

        # Log CHS distribution
        chs_values = [s.chs for s in samples]
        low = sum(1 for c in chs_values if c <= 0.3)
        mid = sum(1 for c in chs_values if 0.3 < c <= 0.6)
        high = sum(1 for c in chs_values if c > 0.6)
        avg_chs = sum(chs_values) / max(len(chs_values), 1)
        logger.info(
            f"CHS distribution: "
            f"good(≤0.3)={low} | moderate(0.3-0.6)={mid} | bad(>0.6)={high} | "
            f"avg={avg_chs:.3f}"
        )
    else:
        logger.info(f"Loading saved samples from {args.samples}")
        samples = TrainingSignalGenerator.load(args.samples)
        logger.info(f"Loaded {len(samples)} samples")

    # Validate samples before training
    valid = [s for s in samples if s.answer and s.answer.strip()]
    if len(valid) < len(samples):
        logger.warning(
            f"Filtered {len(samples) - len(valid)} samples with empty answers"
        )
        samples = valid

    if len(samples) == 0:
        logger.error("No valid training samples! Cannot proceed.")
        sys.exit(1)

    # ── Phase 2: Fine-tune ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 2: QLoRA Fine-Tuning")
    logger.info("=" * 60)

    trainer = MetricGuidedQLoRATrainer(cfg)
    trainer.train(samples)

    output_dir = cfg.get("training", {}).get("output_dir", "outputs/qlora_model")
    logger.info("✓ Training complete")
    logger.info(f"  Adapter saved to: {output_dir}")
    logger.info("  Next steps:")
    logger.info(
        f"    1. Set adapter_path: \"{output_dir}\" in config.yaml"
    )
    logger.info("    2. Run: python scripts/run_inference.py --query 'test query'")
    logger.info("    3. Run: python scripts/evaluate.py --adapter " + output_dir)


if __name__ == "__main__":
    main()