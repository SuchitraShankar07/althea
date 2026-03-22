"""run_inference.py – Run the Althea pipeline on one or more queries.

Usage
-----
    python scripts/run_inference.py \
        --query "What are the symptoms of type 2 diabetes?" \
        --config config/config.yaml \
        [--no_diagnosis] [--evaluate]

    # Or pass a JSON file with multiple queries:
    python scripts/run_inference.py \
        --input_file queries.json \
        --output_file results.json
"""

import argparse
import json
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Althea inference pipeline.")
    parser.add_argument("--query", default=None, help="Single query string.")
    parser.add_argument(
        "--input_file",
        default=None,
        help="JSON file with a list of query strings.",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Path to write JSON results (stdout if omitted).",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--no_diagnosis",
        action="store_true",
        help="Skip hallucination diagnosis.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run RAG quality evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.query is None and args.input_file is None:
        print("Error: provide --query or --input_file.", file=sys.stderr)
        sys.exit(1)

    from src.pipeline import Pipeline

    pipeline = Pipeline(config_path=args.config)

    queries = []
    if args.query:
        queries.append(args.query)
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as fh:
            queries.extend(json.load(fh))

    results = []
    for query in queries:
        print(f"Processing: {query}")
        result = pipeline.run(
            query=query,
            run_diagnosis=not args.no_diagnosis,
            run_evaluation=args.evaluate,
        )
        results.append(result)

    output = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as fh:
            fh.write(output)
        print(f"Results written to {args.output_file}.")
    else:
        print(output)


if __name__ == "__main__":
    main()
