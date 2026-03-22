# Data Directory

Place raw and processed data files here.

## Expected layout

```
data/
├── raw/          # Original source documents (PDFs, JSONs, etc.)
├── processed/    # Cleaned and chunked text ready for indexing
├── index/        # FAISS index files produced by scripts/build_index.py
│   ├── index.faiss
│   └── documents.json
└── checkpoints/  # QLoRA fine-tuned model checkpoints
```

## Notes

- Large binary files (indexes, model weights) should **not** be committed to Git.
  Add them to `.gitignore` or use Git LFS.
- See `scripts/build_index.py` for instructions on how to build the FAISS index
  from the processed text files.
