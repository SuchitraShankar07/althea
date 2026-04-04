# Failure-Aware RAG

**Hallucination diagnosis at the claim level, with metric-guided QLoRA fine-tuning.**

```
Query → Retrieve docs → Generate answer → Extract claims →
Verify each claim via NLI → Compute metrics → Fine-tune generator
```

## Hallucination Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **SCR** — Support Coverage Ratio | supported_claims / total_claims | ↑ better |
| **CR** — Conflict Rate | contradicted_claims / total_claims | ↓ better |
| **TVE** — Temporal Validity Error | outdated_claims / temporal_claims | ↓ better |
| **CDEE** — Cross-Doc Entailment Error | invalid_synthesis / multi_doc_claims | ↓ better |
| **CHS** — Composite Hallucination Score | weighted sum of above | ↓ better |

## Quick Start

```bash
make install        # create .venv, install deps, download spaCy model
make demo           # full pipeline demo — no GPU, no downloads
make demo-verbose   # same + per-claim labels and ASCII metric bars
```

## Workflow

```bash
# 1. Build retrieval index (CPU, ~4 GB RAM, ~10 min for 50k docs)
make index N_DOCS=50000

# 2. Download QA evaluation datasets
make data MAX_SAMPLES=500

# 3. Run a single query with mock generator (CPU-safe)
python scripts/run_inference.py --mock-generate --query "What is DNA?"

# 4. Train (requires CUDA GPU, ≥16 GB VRAM)
make train TRAIN_METHOD=dpo

# 5. Evaluate baseline vs fine-tuned
make evaluate-compare

# 6. Generate charts
make visualise
```

## Common Targets

| Target | Description |
|--------|-------------|
| `make help` | List all available targets |
| `make install` | Install runtime dependencies |
| `make install-dev` | + matplotlib, pytest |
| `make demo` | Run end-to-end mock demo |
| `make test` | Run all tests |
| `make test-unit` | Unit tests only (no ML models) |
| `make test-integration` | Integration tests with mocks |
| `make test-cov` | Coverage report |
| `make clean` | Remove eval/figure outputs |
| `make clean-all` | Remove all generated data |

## Training Strategies

| Strategy | Flag | Description |
|----------|------|-------------|
| DPO | `--method dpo` | Direct Preference Optimisation on ranked answer pairs |
| Rejection SFT | `--method rejection` | Fine-tune only on low-CHS (high-quality) answers |
| Metric Loss | `--method metric_loss` | CE loss + λ × CHS penalty per sample |

## Hardware Requirements

| Task | Minimum |
|------|---------|
| `demo.py` (mock) | CPU only, any RAM |
| `build_index.py` | CPU only, 4 GB RAM |
| `run_inference.py --mock-generate` | CPU only, 4 GB RAM |
| `run_inference.py` with TinyLlama | CPU, 8 GB RAM (slow) |
| `run_inference.py` with Mistral-7B | 16 GB VRAM GPU |
| QLoRA fine-tuning (7B) | 16 GB VRAM GPU |

## Project Structure

```
├── config/config.yaml          # All hyperparameters
├── src/
│   ├── retrieval/              # DenseEncoder + FAISSIndex + Retriever
│   ├── generation/             # RAGGenerator (4-bit QLoRA, CPU guard)
│   ├── diagnosis/              # ClaimExtractor + NLI + MetricEngine
│   ├── evaluation/             # F1/EM + AggregateResult + compare()
│   ├── training/               # MetricGuidedQLoRATrainer (DPO/rejection/metric_loss)
│   └── pipeline.py             # FailureAwareRAGPipeline
├── scripts/
│   ├── demo.py                 # Zero-setup end-to-end demo
│   ├── run_inference.py        # Single query / batch (--mock-generate flag)
│   ├── build_index.py          # Download Wikipedia + build FAISS index
│   ├── prepare_data.py         # Download HotpotQA / NQ / PopQA / RAGTruth
│   ├── train.py                # Collect samples + QLoRA fine-tune
│   ├── evaluate.py             # Baseline vs fine-tuned comparison
│   └── visualise_results.py    # 6 matplotlib charts
├── tests/
│   ├── conftest.py             # ML stubs + fixtures (no GPU needed)
│   ├── test_pipeline_units.py  # Unit tests
│   └── test_integration.py     # Integration tests with mocks
└── notebooks/exploration.ipynb # Interactive walkthrough
```

## Notes

- `.cache/` stores model cache; `data/` and `outputs/` store generated artefacts.
- `make train` requires a CUDA GPU.
- The generator raises `RuntimeError` immediately for large models (Mistral, LLaMA 7B+) on CPU — use `--mock-generate` or TinyLlama for CPU testing.
- On Python ≥ 3.14, spaCy may fail to load; the system falls back to regex sentence splitting automatically.
