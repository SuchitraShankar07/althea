#!/usr/bin/env python
"""
scripts/evaluate.py
Evaluate baseline vs fine-tuned model on a QA dataset.
Computes both QA metrics (F1, EM) and hallucination metrics.

Usage:
    # Evaluate baseline only
    python scripts/evaluate.py --config config/config.yaml --dataset hotpotqa

    # Compare baseline vs fine-tuned adapter
    python scripts/evaluate.py \
        --config config/config.yaml \
        --dataset hotpotqa \
        --adapter outputs/qlora/dpo \
        --max-samples 200
"""

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from loguru import logger

from src.diagnosis.metric_engine import HallucinationMetrics
from src.evaluation.evaluator import RAGEvaluator
from src.pipeline import FailureAwareRAGPipeline


# ── Dataset loaders ───────────────────────────────────────────────────────────
def load_hotpotqa(split: str = "validation", max_samples: int = 200):
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split=split, streaming=True)
    queries, answers = [], []
    for row in ds:
        queries.append(row["question"])
        answers.append(row["answer"])
        if len(queries) >= max_samples:
            break
    return queries, answers


def load_natural_questions(split: str = "validation", max_samples: int = 200):
    from datasets import load_dataset
    ds = load_dataset("natural_questions", split=split, streaming=True)
    queries, answers = [], []
    for row in ds:
        short_answers = row["annotations"]["short_answers"]
        if not short_answers or not short_answers[0]["text"]:
            continue
        queries.append(row["question"]["text"])
        answers.append(short_answers[0]["text"][0])
        if len(queries) >= max_samples:
            break
    return queries, answers


def load_popqa(max_samples: int = 200):
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split="test", streaming=True)
    queries, answers = [], []
    for row in ds:
        queries.append(row["question"])
        answers.append(row["possible_answers"][0] if row["possible_answers"] else "")
        if len(queries) >= max_samples:
            break
    return queries, answers


DATASET_LOADERS = {
    "hotpotqa": load_hotpotqa,
    "natural_questions": load_natural_questions,
    "popqa": load_popqa,
}


# ── Evaluation runner ─────────────────────────────────────────────────────────
def evaluate_model(pipeline, queries, ground_truths, tag):
    """Run inference + diagnosis and return aggregate results."""
    predictions, hal_metrics = [], []

    for i, (q, gt) in enumerate(zip(queries, ground_truths)):
        logger.info(f"[{i+1}/{len(queries)}] {q[:60]}")
        result = pipeline.run_inference(q, diagnose=True)
        predictions.append(result["answer"])
        m = result["diagnosis"]["metrics"]
        hal_metrics.append(HallucinationMetrics(**m))

    return pipeline.evaluator.evaluate(
        queries, predictions, ground_truths, hal_metrics, model_tag=tag
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", default="hotpotqa", choices=list(DATASET_LOADERS))
    parser.add_argument("--adapter", default=None, help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/eval")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load dataset
    loader = DATASET_LOADERS[args.dataset]
    logger.info(f"Loading {args.dataset} (max {args.max_samples} samples)...")
    queries, ground_truths = loader(max_samples=args.max_samples)
    logger.info(f"Loaded {len(queries)} samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = RAGEvaluator(results_dir=str(output_dir))

    # ── Baseline evaluation ───────────────────────────────────
    logger.info("=== Evaluating Baseline ===")
    cfg["generation"]["adapter_path"] = None
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        baseline_config_path = tmp.name
    baseline_pipeline = FailureAwareRAGPipeline.from_config(baseline_config_path)
    baseline_pipeline.evaluator = evaluator
    baseline_result = evaluate_model(baseline_pipeline, queries, ground_truths, tag="baseline")
    Path(baseline_config_path).unlink(missing_ok=True)

    # ── Fine-tuned evaluation ─────────────────────────────────
    if args.adapter:
        logger.info(f"=== Evaluating Fine-tuned Model ({args.adapter}) ===")
        cfg_tuned = copy.deepcopy(cfg)
        cfg_tuned["generation"]["adapter_path"] = args.adapter

        # Re-build pipeline with adapter
        from src.generation.generator import RAGGenerator
        from src.retrieval.retriever import Retriever
        from src.diagnosis.diagnose import HallucinationDiagnoser

        retriever = Retriever.from_config(cfg_tuned["retrieval"])
        generator = RAGGenerator.from_config(cfg_tuned["generation"])
        diagnoser = HallucinationDiagnoser.from_config(cfg_tuned, retriever)
        tuned_pipeline = FailureAwareRAGPipeline(
            retriever=retriever,
            generator=generator,
            diagnoser=diagnoser,
            evaluator=evaluator,
            cfg=cfg_tuned,
        )
        tuned_result = evaluate_model(tuned_pipeline, queries, ground_truths, tag="tuned")

        # ── Comparison ────────────────────────────────────────
        logger.info("=== Baseline vs Fine-tuned Comparison ===")
        delta = evaluator.compare(baseline_result, tuned_result)

        comparison_path = Path(args.output_dir) / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump({
                "baseline": vars(baseline_result),
                "tuned": vars(tuned_result),
                "delta": delta,
            }, f, indent=2)
        logger.info(f"Comparison saved to {comparison_path}")
    else:
        logger.info("No adapter provided — baseline only evaluation complete")
        logger.info(baseline_result.summary())


if __name__ == "__main__":
    main()
