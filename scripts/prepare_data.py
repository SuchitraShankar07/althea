#!/usr/bin/env python3
"""
Download and reformat datasets into unified JSONL files for this pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def load_hotpotqa(split: str = "validation", max_samples: int = 1000) -> Iterator[dict]:
    from datasets import load_dataset

    logger.info(f"Loading HotpotQA ({split}, max={max_samples})")
    ds = load_dataset("hotpot_qa", "distractor", split=split, streaming=True, trust_remote_code=True)
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        yield {
            "id": row["id"],
            "query": row["question"],
            "answer": row["answer"],
            "source": "hotpotqa",
            "type": row.get("type", ""),
            "level": row.get("level", ""),
        }


def load_natural_questions(split: str = "validation", max_samples: int = 1000) -> Iterator[dict]:
    from datasets import load_dataset

    logger.info(f"Loading Natural Questions ({split}, max={max_samples})")
    ds = load_dataset("natural_questions", split=split, streaming=True, trust_remote_code=True)
    idx = 0
    for row in ds:
        if idx >= max_samples:
            break
        short = row["annotations"]["short_answers"]
        if not short or not short[0]["text"]:
            continue
        yield {
            "id": str(row["id"]),
            "query": row["question"]["text"],
            "answer": short[0]["text"][0],
            "source": "natural_questions",
        }
        idx += 1


def load_popqa(split: str = "test", max_samples: int = 1000) -> Iterator[dict]:
    from datasets import load_dataset

    # PopQA is typically used with test split.
    use_split = "test"
    logger.info(f"Loading PopQA ({use_split}, max={max_samples})")
    ds = load_dataset("akariasai/PopQA", split=use_split, streaming=True, trust_remote_code=True)
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        answers = row.get("possible_answers", [])
        yield {
            "id": str(row.get("id", i)),
            "query": row["question"],
            "answer": answers[0] if answers else "",
            "answer_aliases": answers,
            "source": "popqa",
            "prop": row.get("prop", ""),
        }


def load_triviaqa(split: str = "validation", max_samples: int = 1000) -> Iterator[dict]:
    from datasets import load_dataset

    logger.info(f"Loading TriviaQA ({split}, max={max_samples})")
    ds = load_dataset("trivia_qa", "rc", split=split, streaming=True, trust_remote_code=True)
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        yield {
            "id": row["question_id"],
            "query": row["question"],
            "answer": row["answer"]["value"],
            "answer_aliases": row["answer"].get("aliases", []),
            "source": "triviaqa",
        }


def load_ragtruth(split: str = "test", max_samples: int = 500) -> Iterator[dict]:
    from datasets import load_dataset

    logger.info(f"Loading RAGTruth ({split}, max={max_samples})")
    try:
        ds = load_dataset("wandb/RAGTruth", split=split, streaming=True, trust_remote_code=True)
    except Exception:
        ds = load_dataset("RAGTruth/RAGTruth", split=split, streaming=True, trust_remote_code=True)

    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        yield {
            "id": str(row.get("id", i)),
            "query": row.get("question", row.get("query", "")),
            "answer": row.get("response", row.get("answer", "")),
            "reference": row.get("reference", ""),
            "hallucination_label": row.get("hallucination", row.get("label", None)),
            "source": "ragtruth",
        }


DATASET_LOADERS = {
    "hotpotqa": load_hotpotqa,
    "nq": load_natural_questions,
    "popqa": load_popqa,
    "triviaqa": load_triviaqa,
    "ragtruth": load_ragtruth,
}


def prepare_wikipedia_corpus(output_path: str, n_docs: int = 100_000) -> None:
    from datasets import load_dataset

    logger.info(f"Downloading Wikipedia ({n_docs:,} docs)...")
    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for row in ds:
            if count >= n_docs:
                break
            words = row["text"].split()
            for j in range(0, min(len(words), 600), 200):
                passage = " ".join(words[j : j + 200])
                if len(passage.split()) < 30:
                    continue
                doc = {
                    "id": f"{row['id']}_p{j//200}",
                    "title": row["title"],
                    "text": passage,
                }
                f.write(json.dumps(doc) + "\n")
                count += 1
                if count >= n_docs:
                    break
    logger.info(f"Saved {count:,} passages to {output_path}")


def extract_training_queries(input_paths: list[str], output_path: str, max_total: int = 5000) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w") as out:
        for path in input_paths:
            with open(path) as f:
                for line in f:
                    if written >= max_total:
                        break
                    row = json.loads(line)
                    if row.get("query") and row.get("answer"):
                        out.write(json.dumps({"query": row["query"], "answer": row["answer"]}) + "\n")
                        written += 1
    logger.info(f"Wrote {written} training queries to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_LOADERS) + ["all", "corpus"],
        default=["hotpotqa"],
        help="Which datasets to download",
    )
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=500, help="Samples per dataset")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--corpus-docs", type=int, default=50_000, help="Wikipedia passages for corpus")
    parser.add_argument("--merge-train", action="store_true", help="Merge QA splits into train_queries.jsonl")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_to_run = list(DATASET_LOADERS.keys()) if "all" in args.datasets else args.datasets

    saved_paths = []
    for name in datasets_to_run:
        if name == "corpus":
            continue
        loader = DATASET_LOADERS[name]
        out_path = output_dir / f"{name}_{args.split}.jsonl"

        if out_path.exists():
            logger.info(f"Skipping {name} — {out_path} already exists")
            saved_paths.append(str(out_path))
            continue

        try:
            records = list(loader(split=args.split, max_samples=args.max_samples))
            with open(out_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            logger.info(f"{name}: {len(records)} records -> {out_path}")
            saved_paths.append(str(out_path))
        except Exception as e:
            logger.warning(f"{name}: failed ({e})")

    if "corpus" in args.datasets or "all" in args.datasets:
        corpus_path = output_dir / "corpus.jsonl"
        if not corpus_path.exists():
            prepare_wikipedia_corpus(str(corpus_path), n_docs=args.corpus_docs)
        else:
            logger.info(f"Corpus already exists at {corpus_path}")

    if args.merge_train and saved_paths:
        train_path = str(output_dir / "train_queries.jsonl")
        extract_training_queries(saved_paths, train_path)

    logger.info(f"Data preparation complete. Files in: {output_dir}")
    for p in list(output_dir.glob("*.jsonl")):
        lines = sum(1 for _ in open(p))
        size_kb = p.stat().st_size / 1024
        logger.info(f"{p.name:<40} {lines:>6} records ({size_kb:>7.1f} KB)")


if __name__ == "__main__":
    main()
