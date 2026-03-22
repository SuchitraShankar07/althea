"""build_index.py – Build a FAISS index from a directory of text documents.

Usage
-----
    python scripts/build_index.py \
        --input_dir data/processed \
        --output_dir data/index \
        --config config/config.yaml
"""

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from documents.")
    parser.add_argument(
        "--input_dir",
        default="data/processed",
        help="Directory containing .txt or .json document files.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/index",
        help="Directory where the FAISS index will be saved.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml.",
    )
    return parser.parse_args()


def load_documents(input_dir: str):
    """Load documents from *input_dir*.

    Supports plain ``.txt`` files and ``.json`` files containing a list of
    ``{"id": ..., "text": ...}`` dicts.
    """
    documents = []
    for fname in sorted(os.listdir(input_dir)):
        fpath = os.path.join(input_dir, fname)
        if fname.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as fh:
                documents.append({"id": fname, "text": fh.read()})
        elif fname.endswith(".json"):
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    documents.extend(data)
                else:
                    documents.append(data)
    return documents


def main():
    import numpy as np
    import yaml
    import faiss

    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    retrieval_cfg = config.get("retrieval", {})

    from src.retrieval import Encoder

    encoder = Encoder(model_name=retrieval_cfg.get("model_name"))

    print(f"Loading documents from {args.input_dir} …")
    documents = load_documents(args.input_dir)
    if not documents:
        raise ValueError(f"No documents found in {args.input_dir}")
    print(f"  Loaded {len(documents)} documents.")

    texts = [d["text"] for d in documents]
    print("Encoding documents …")
    embeddings = encoder.encode(texts, show_progress_bar=True).astype(np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(args.output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(args.output_dir, "index.faiss"))
    with open(os.path.join(args.output_dir, "documents.json"), "w", encoding="utf-8") as fh:
        json.dump(documents, fh, ensure_ascii=False, indent=2)

    print(f"Index saved to {args.output_dir} ({len(documents)} vectors, dim={dimension}).")


if __name__ == "__main__":
    main()
