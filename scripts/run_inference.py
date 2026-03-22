#!/usr/bin/env python
"""
scripts/run_inference.py
Run the full RAG + diagnosis pipeline on one query or a JSONL query file.

Usage:
    # Single query
    python scripts/run_inference.py --query "What caused the 2008 financial crisis?"

    # Batch from file  (JSONL: {"query": "...", "answer": "..."} per line)
    python scripts/run_inference.py --input data/queries.jsonl --output outputs/predictions.jsonl

    # Skip diagnosis for faster inference
    python scripts/run_inference.py --query "..." --no-diagnose
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.pipeline import FailureAwareRAGPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--query", default=None, help="Single query string")
    parser.add_argument("--input", default=None, help="JSONL file with queries")
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    parser.add_argument("--no-diagnose", action="store_true")
    args = parser.parse_args()

    pipeline = FailureAwareRAGPipeline.from_config(args.config)
    diagnose = not args.no_diagnose

    if args.query:
        # ── Single query ──────────────────────────────────────
        result = pipeline.run_inference(args.query, diagnose=diagnose)
        print("\n" + "=" * 60)
        print(f"QUERY:  {result['query']}")
        print(f"ANSWER: {result['answer']}")
        if diagnose and "diagnosis" in result:
            print(f"\nDIAGNOSIS:")
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