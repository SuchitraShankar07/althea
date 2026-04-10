#!/usr/bin/env python
"""
scripts/run_inference.py
Run the full RAG + diagnosis pipeline on one query or a JSONL query file.

Usage:
    # Single query (requires GPU for large models)
    python scripts/run_inference.py --query "What caused the 2008 financial crisis?"

    # CPU-safe mock: retrieval + NLI diagnosis, no LLM download
    python scripts/run_inference.py --mock-generate --query "What is DNA?"

    # Batch from file  (JSONL: {"query": "...", "answer": "..."} per line)
    python scripts/run_inference.py --input data/queries.jsonl --output outputs/predictions.jsonl

    # Skip diagnosis for faster inference
    python scripts/run_inference.py --query "..." --no-diagnose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


# ── Mock generator for CPU-safe testing ──────────────────────────────────────
class _MockGenerator:
    """
    Lightweight stand-in for RAGGenerator.
    Splits the top retrieved document's text into sentences and returns
    the first two as the 'answer'.  No model download required.
    """

    def generate(self, query: str, documents: List[dict]) -> str:
        if not documents:
            return "No documents retrieved."
        import re
        top_text = documents[0]["text"]
        # Split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", top_text.strip())
        answer = " ".join(s.strip() for s in sentences[:2] if s.strip())
        return answer or top_text[:200]

    def generate_batch(self, queries, docs_list):
        return [self.generate(q, d) for q, d in zip(queries, docs_list)]


# ── Pipeline builder ──────────────────────────────────────────────────────────
def _build_pipeline(config_path: str, mock_generate: bool):
    """
    Build the pipeline.  If mock_generate is True, substitute a _MockGenerator
    so the full retrieval + NLI stack runs without any LLM.
    """
    from src.pipeline import FailureAwareRAGPipeline

    if not mock_generate:
        return FailureAwareRAGPipeline.from_config(config_path)

    # Build pipeline normally then swap the generator
    import yaml
    from src.retrieval.retriever import Retriever
    from src.diagnosis.diagnose import HallucinationDiagnoser
    from src.evaluation.evaluator import RAGEvaluator

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    logger.info("Mock-generate mode: skipping LLM load")
    retriever = Retriever.from_config(cfg["retrieval"])
    diagnoser = HallucinationDiagnoser.from_config(cfg, retriever)
    evaluator = RAGEvaluator(
        results_dir=cfg.get("evaluation", {}).get("results_dir", "outputs/eval")
    )
    pipeline = FailureAwareRAGPipeline(
        retriever=retriever,
        generator=_MockGenerator(),   # type: ignore[arg-type]
        diagnoser=diagnoser,
        evaluator=evaluator,
        cfg=cfg,
    )
    return pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run Failure-Aware RAG inference (single query or batch)."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--query", default=None, help="Single query string")
    parser.add_argument("--input", default=None, help="JSONL file with queries")
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    parser.add_argument(
        "--mock-generate",
        action="store_true",
        help=(
            "Use a lightweight mock generator (no LLM download). "
            "Retrieval + NLI claim diagnosis still run for real."
        ),
    )
    parser.add_argument("--no-diagnose", action="store_true")
    args = parser.parse_args()

    pipeline = _build_pipeline(args.config, mock_generate=args.mock_generate)
    diagnose = not args.no_diagnose

    if args.query:
        # ── Single query ──────────────────────────────────────
        result = pipeline.run_inference(args.query, diagnose=diagnose)
        print("\n" + "=" * 60)
        print(f"QUERY:  {result['query']}")
        print(f"ANSWER: {result['answer']}")
        if diagnose and "diagnosis" in result:
            print("\nDIAGNOSIS:")
            print(f"  {result['diagnosis']['summary']}")
            print(f"  Claims found: {len(result['diagnosis']['claims'])}")
            for i, claim in enumerate(result['diagnosis']['claims'], 1):
                print(f"    {i}. {claim}")
        print("=" * 60)

    elif args.input:
        # ── Batch mode ────────────────────────────────────────
        queries, ground_truths = [], []
        with open(args.input) as f:
            for line in f:
                row = json.loads(line.strip())
                queries.append(row["query"])
                ground_truths.append(row.get("answer", ""))

        results = pipeline.run_batch(
            queries,
            ground_truths=ground_truths if any(ground_truths) else None,
            diagnose=diagnose,
        )

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for r in results:
                # Serialise documents (strip embedding vectors if present)
                docs = [{"doc_id": d["doc_id"], "text": d["text"][:200]} for d in r["documents"]]
                row = {
                    "query": r["query"],
                    "answer": r["answer"],
                    "documents": docs,
                }
                if diagnose and "diagnosis" in r:
                    row["metrics"] = r["diagnosis"]["metrics"]
                f.write(json.dumps(row) + "\n")

        logger.info(f"Saved {len(results)} predictions to {args.output}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()