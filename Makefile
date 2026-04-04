PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

CONFIG ?= config/config.yaml
MODEL_NAME ?= phi3.5 by Microsoft
DATA_DIR ?= data
OUTPUT_DIR ?= outputs
MAX_SAMPLES ?= 500
N_DOCS ?= 50000
TRAIN_METHOD ?= dpo

QUERY ?= What caused the 2008 financial crisis?
DATASET ?= hotpotqa
ADAPTER ?= $(OUTPUT_DIR)/qlora/$(TRAIN_METHOD)
QUERIES ?= $(DATA_DIR)/train_queries.jsonl
MAX_EVAL_SAMPLES ?= 100

.DEFAULT_GOAL := help

.PHONY: help dirs venv install install-dev spacy \
	demo demo-verbose demo-save \
	index data data-all query batch \
	evaluate evaluate-compare train train-dpo train-rejection train-metric-loss train-from-samples \
	visualise visualise-training \
	test test-unit test-integration test-cov \
	run pipeline clean clean-all

help:
	@echo ""
	@echo "  Failure-Aware RAG — available targets"
	@echo "  ------------------------------------------------------"
	@echo "  install            Create .venv and install runtime deps"
	@echo "  install-dev        Install runtime + dev deps"
	@echo "  spacy              Download spaCy model (if compatible)"
	@echo ""
	@echo "  demo               Run end-to-end mock demo (no GPU)"
	@echo "  index              Build FAISS retrieval index"
	@echo "  data               Download/prepare QA datasets"
	@echo "  evaluate           Baseline evaluation"
	@echo "  evaluate-compare   Baseline vs fine-tuned adapter"
	@echo "  train              Collect data + QLoRA train (CUDA required)"
	@echo "  visualise          Generate evaluation figures"
	@echo ""
	@echo "  run                Run your 5-step flow with safe skips"
	@echo "  pipeline           Full data->train->evaluate->visualise pipeline"
	@echo ""
	@echo "  test               Run all tests"
	@echo "  clean              Remove generated outputs"
	@echo ""

dirs:
	mkdir -p $(DATA_DIR) $(OUTPUT_DIR) .cache

$(VENV)/bin/python:
	@if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "Error: $(PYTHON) not found. Set PYTHON=<interpreter>."; \
		exit 1; \
	fi
	$(PYTHON) -m venv $(VENV)

venv: $(VENV)/bin/python

install: dirs venv
	$(PIP) install -r requirements.txt
	@if $(PY) -c "import sys; raise SystemExit(0 if sys.version_info < (3,14) else 1)"; then \
		echo "Installing spaCy model en_core_web_sm..."; \
		$(PY) -m spacy download en_core_web_sm; \
	else \
		echo "Skipping spaCy model download on Python >= 3.14 (spaCy compatibility issue)."; \
	fi

install-dev: install
	$(PIP) install matplotlib pytest pytest-cov

spacy: venv
	$(PY) -m spacy download en_core_web_sm

demo: install
	$(PY) scripts/demo.py

demo-verbose: install
	$(PY) scripts/demo.py --verbose

demo-save: install
	$(PY) scripts/demo.py --save $(OUTPUT_DIR)/demo_results.jsonl

index: install
	$(PY) scripts/build_index.py --config $(CONFIG) --n_docs $(N_DOCS)

data: install
	$(PY) scripts/prepare_data.py --datasets hotpotqa nq popqa --max-samples $(MAX_SAMPLES) --output-dir $(DATA_DIR) --merge-train

data-all: install
	$(PY) scripts/prepare_data.py --datasets all --max-samples $(MAX_SAMPLES) --output-dir $(DATA_DIR) --corpus-docs $(N_DOCS) --merge-train

query: install
	$(PY) scripts/run_inference.py --config $(CONFIG) --query "$(QUERY)"

batch: install
	$(PY) scripts/run_inference.py --config $(CONFIG) --input $(DATA_DIR)/hotpotqa_validation.jsonl --output $(OUTPUT_DIR)/predictions.jsonl

evaluate: install
	$(PY) scripts/evaluate.py --config $(CONFIG) --dataset $(DATASET) --max-samples $(MAX_EVAL_SAMPLES) --output-dir $(OUTPUT_DIR)/eval

evaluate-compare: install
	@if [ ! -d "$(ADAPTER)" ]; then \
		echo "Error: adapter not found at $(ADAPTER)"; \
		exit 1; \
	fi
	$(PY) scripts/evaluate.py --config $(CONFIG) --dataset $(DATASET) --adapter $(ADAPTER) --max-samples $(MAX_EVAL_SAMPLES) --output-dir $(OUTPUT_DIR)/eval

train: install
	@if [ ! -f "$(QUERIES)" ]; then \
		echo "Error: training queries file not found at $(QUERIES)"; \
		exit 1; \
	fi
	@if ! $(PY) -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then \
		echo "Error: training requires a CUDA GPU."; \
		exit 1; \
	fi
	$(PY) scripts/train.py --config $(CONFIG) --queries $(QUERIES) --method $(TRAIN_METHOD)

train-dpo:
	$(MAKE) train TRAIN_METHOD=dpo

train-rejection:
	$(MAKE) train TRAIN_METHOD=rejection

train-metric-loss:
	$(MAKE) train TRAIN_METHOD=metric_loss

train-from-samples: install
	@if ! $(PY) -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then \
		echo "Error: training requires a CUDA GPU."; \
		exit 1; \
	fi
	$(PY) scripts/train.py --config $(CONFIG) --samples $(OUTPUT_DIR)/training_samples.jsonl --method $(TRAIN_METHOD)

visualise: install-dev
	$(PY) scripts/visualise_results.py --results-dir $(OUTPUT_DIR)/eval --samples $(OUTPUT_DIR)/training_samples.jsonl --figures-dir $(OUTPUT_DIR)/figures

visualise-training: install-dev
	$(PY) scripts/visualise_results.py --samples $(OUTPUT_DIR)/training_samples.jsonl --charts training chs --figures-dir $(OUTPUT_DIR)/figures

test: dirs venv
	$(PIP) install pytest
	$(PY) -m pytest tests/ -v --tb=short

test-unit: dirs venv
	$(PIP) install pytest
	$(PY) -m pytest tests/test_pipeline_units.py -v --tb=short

test-integration: dirs venv
	$(PIP) install pytest
	$(PY) -m pytest tests/test_integration.py -v --tb=short

test-cov: dirs venv
	$(PIP) install pytest pytest-cov
	$(PY) -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html:$(OUTPUT_DIR)/coverage

run: install
	@echo "Step 2/5: Build index"
	$(MAKE) index CONFIG=$(CONFIG) N_DOCS=$(N_DOCS)
	@echo "Step 3/5: Run inference"
	$(MAKE) query CONFIG=$(CONFIG) QUERY="$(QUERY)"
	@echo "Step 4/5: Train (GPU required)"
	@if [ -f "$(QUERIES)" ] && $(PY) -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then \
		$(MAKE) train CONFIG=$(CONFIG) QUERIES=$(QUERIES) TRAIN_METHOD=$(TRAIN_METHOD); \
	else \
		echo "Skipping training: missing $(QUERIES) or no CUDA GPU."; \
	fi
	@echo "Step 5/5: Evaluate"
	@if [ -d "$(ADAPTER)" ]; then \
		$(MAKE) evaluate-compare CONFIG=$(CONFIG) DATASET=$(DATASET) ADAPTER=$(ADAPTER) MAX_EVAL_SAMPLES=$(MAX_EVAL_SAMPLES); \
	else \
		echo "Adapter not found at $(ADAPTER); running baseline-only evaluation."; \
		$(MAKE) evaluate CONFIG=$(CONFIG) DATASET=$(DATASET) MAX_EVAL_SAMPLES=$(MAX_EVAL_SAMPLES); \
	fi

pipeline: install data index train evaluate-compare visualise
	@echo ""
	@echo "Pipeline complete."
	@echo "Results: $(OUTPUT_DIR)/eval/"
	@echo "Figures: $(OUTPUT_DIR)/figures/"
	@echo ""

clean:
	rm -rf $(OUTPUT_DIR)/eval/
	rm -rf $(OUTPUT_DIR)/figures/
	rm -rf $(OUTPUT_DIR)/training_samples.jsonl
	rm -rf $(OUTPUT_DIR)/predictions.jsonl

clean-all: clean
	rm -rf $(OUTPUT_DIR)/
	rm -rf $(DATA_DIR)/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
