# RAG Implementation Deep Dive

## 1) System Scope and Design Intent

This repository implements a failure-aware Retrieval-Augmented Generation (RAG) system where the generation stage is explicitly audited at claim-level, then turned into a training signal for iterative improvement. The project objective is not only answer quality, but calibrated factuality under retrieval constraints.

High-level system statement from repository docs:
- Claim-level hallucination diagnosis and metric-guided QLoRA fine-tuning are central, not optional extras.
- See project overview and workflow in `README.md`.

Implementation anchors:
- `README.md:1`
- `README.md:28`
- `src/pipeline.py:36`

---

## 2) End-to-End Dataflow (Operational Graph)

### 2.1 Primary graph

1. Query encoding and dense retrieval.
2. Prompt assembly from retrieved passages.
3. LLM answer generation.
4. Answer decomposition into factual claims.
5. Per-claim evidence retrieval and NLI verification.
6. Hallucination metric aggregation.
7. Hallucination taxonomy scoring and labeling.
8. Optional training signal emission.

Pipeline entry points:
- Single inference: `src/pipeline.py:60`
- Batch inference and optional evaluator call: `src/pipeline.py:88`
- Training sample collection: `src/pipeline.py:116`
- Fine-tuning trigger: `src/pipeline.py:139`

### 2.2 Factory-wired composition

`FailureAwareRAGPipeline.from_config` creates and wires all major subsystems from YAML config:
- Retriever
- Generator
- Diagnoser
- Evaluator

Anchor:
- `src/pipeline.py:154`

---

## 3) Retrieval Subsystem Theory and Implementation

## 3.1 Representation theory

The retriever uses sentence-transformer embeddings and FAISS inner-product search over normalized vectors. If embeddings are $L_2$-normalized, maximizing inner product approximates maximizing cosine similarity.

\[
\operatorname{cos}(u,v) = \frac{u^T v}{\|u\|\|v\|} \quad \Rightarrow \quad
\text{for } \|u\|=\|v\|=1,\; \operatorname{cos}(u,v)=u^T v
\]

Implementation anchors:
- Encoder abstraction: `src/retrieval/encoder.py:22`
- Encoding call with normalization option: `src/retrieval/encoder.py:30`
- FAISS flat IP index: `src/retrieval/encoder.py:48`
- IP search interface: `src/retrieval/encoder.py:80`

## 3.2 Index lifecycle

1. Parse JSONL corpus.
2. Batch encode documents.
3. Add vectors and metadata to index.
4. Persist FAISS index plus metadata sidecar.

Anchors:
- Build from corpus: `src/retrieval/encoder.py:138`
- Add documents: `src/retrieval/encoder.py:62`
- Save index and metadata: `src/retrieval/encoder.py:105`
- Reload persisted index: `src/retrieval/encoder.py:123`
- Build script entry: `scripts/build_index.py:91`
- Wikipedia/data fallback logic: `scripts/build_index.py:30`

## 3.3 Retrieval API behavior

Retriever serves query-time and batch retrieval:
- Single query: `src/retrieval/retriever.py:35`
- Batch query: `src/retrieval/retriever.py:41`
- Config bootstrap and index existence requirement: `src/retrieval/retriever.py:48`

The default top-k policy comes from config:
- `config/config.yaml:8`

---

## 4) Generation Subsystem Theory and Implementation

## 4.1 Prompt grounding strategy

Prompting follows strict grounded-answer instructions:
- The model is told to use only provided context and to abstain when insufficient.

Anchors:
- System instruction constant: `src/generation/generator.py:22`
- Prompt construction function: `src/generation/generator.py:28`

## 4.2 Inference model loading and hardware policy

`RAGGenerator` supports optional 4-bit quantized loading and LoRA adapter injection. It also includes explicit CPU guardrails for large models (7B+ heuristics), forcing safe failure instead of silent thrashing.

Anchors:
- Generator class: `src/generation/generator.py:44`
- Large-model detector: `src/generation/generator.py:60`
- Runtime generation path: `src/generation/generator.py:152`
- Batch generation wrapper: `src/generation/generator.py:170`
- Config factory: `src/generation/generator.py:177`

Related config controls:
- Model id: `config/config.yaml:24`
- 4-bit toggle: `config/config.yaml:25`
- Adapter path: `config/config.yaml:30`

## 4.3 CPU-safe operational fallback

For environments without GPU or large-model support:
- `scripts/run_inference.py` offers `_MockGenerator` preserving retrieval+diagnosis behavior.
- This allows validating non-LLM stages without full model load.

Anchors:
- Mock generator: `scripts/run_inference.py:34`
- Mock pipeline builder: `scripts/run_inference.py:56`
- CLI flag: `scripts/run_inference.py:100`

---

## 5) Diagnosis Layer: Hallucination as Structured Verification

## 5.1 Why claim decomposition matters

Answers are decomposed into atomic claims to avoid single scalar judgement on an entire long-form response. This supports finer control of factuality signals and targeted penalties.

Anchors:
- Spacy/regex extractor class: `src/diagnosis/claim_extractor.py:19`
- Extract function: `src/diagnosis/claim_extractor.py:38`
- Fallback splitter: `src/diagnosis/claim_extractor.py:63`
- Optional LLM extractor: `src/diagnosis/claim_extractor.py:81`

## 5.2 Verification engine (NLI)

Each claim is checked against retrieved evidence using cross-encoder NLI. Score fusion follows strongest-signal aggregation across evidence candidates, then thresholded into entailment/contradiction/neutral.

Anchors:
- NLI labels: `src/diagnosis/verification_engine.py:26`
- Verification result structure: `src/diagnosis/verification_engine.py:33`
- Verification engine: `src/diagnosis/verification_engine.py:68`
- Core claim verification: `src/diagnosis/verification_engine.py:112`
- Batch verify: `src/diagnosis/verification_engine.py:175`

## 5.3 Temporal reasoning augmentation

Temporal claims are pattern-detected (keywords and explicit years) and flagged as outdated when not entailed by evidence.

Anchors:
- Temporal keyword regex: `src/diagnosis/verification_engine.py:53`
- Year pattern: `src/diagnosis/verification_engine.py:60`
- Temporal claim detector: `src/diagnosis/verification_engine.py:63`

## 5.4 Refusal-aware policy

The diagnoser has an abstention/refusal detector. If the model refuses due to insufficient context, it assigns zero hallucination penalty rather than penalizing abstention.

Anchors:
- Refusal regex: `src/diagnosis/diagnose.py:26`
- Refusal gate: `src/diagnosis/diagnose.py:35`
- Main diagnose path: `src/diagnosis/diagnose.py:74`
- Integration test for refusal behavior: `tests/test_integration.py:114`

---

## 6) Metric Layer and Objective Framing

The project computes four decomposition metrics and one composite score:

- SCR: Support Coverage Ratio
- CR: Conflict Rate
- TVE: Temporal Validity Error
- CDEE: Cross-Document Entailment Error
- CHS: Composite Hallucination Score

Extended taxonomy-level hallucination metrics:
- retrieval_conflict
- overgeneralization
- outdated_information
- synthesis_error
- overall_hallucination_score (OHS)

Definitions in implementation:
- Metric container: `src/diagnosis/metric_engine.py:23`
- Metric computation engine: `src/diagnosis/metric_engine.py:51`
- Compute implementation: `src/diagnosis/metric_engine.py:72`

Composite objective in code form:
\[
\text{CHS} = \lambda \cdot\left[w_{scr}(1-SCR)+w_{cr}CR+w_{tve}TVE+w_{cdee}CDEE\right]
\]

Taxonomy weighted aggregate used for analysis and logging:

\[
	ext{OHS} = 0.3\cdot RC + 0.25\cdot SE + 0.25\cdot OG + 0.2\cdot OI
\]

Weight source:
- `config/config.yaml:44`

Taxonomy detector anchors:
- `src/diagnosis/metric_engine.py:123`
- `src/diagnosis/metric_engine.py:165`
- `src/diagnosis/metric_engine.py:195`
- `src/diagnosis/metric_engine.py:230`

README metric narrative:
- `README.md:10`

---

## 7) Pipeline Modes and Runtime Interfaces

## 7.1 Script-level modes

- Single query / batch inference: `scripts/run_inference.py:91`
- Zero-setup demonstration pipeline: `scripts/demo.py:279`
- Train with collection + fine-tune phases: `scripts/train.py:71`
- Evaluation baseline vs tuned: `scripts/evaluate.py:100`

Optional non-breaking taxonomy toggle:
- `enable_hallucination_eval=True` in pipeline paths.
- CLI disable flag: `--disable-hallucination-eval` for inference/evaluation/training scripts.

## 7.2 Makefile orchestration

Canonical targets:
- Install env: `Makefile:64`
- Demo: `Makefile:79`
- Build index: `Makefile:88`
- Train: `Makefile:113`
- Evaluate baseline: `Makefile:103`
- Evaluate compare: `Makefile:106`
- Visualize metrics: `Makefile:140`

---

## 8) Additional Notes on Legacy/Alternate Components

The repository also contains `src/diagnosis/evidence_retriever.py`, which appears to implement an alternate evidence retrieval strategy via `VectorStore` and repeated sentence-transformer calls. Current primary pipeline path (`src/pipeline.py`) does not instantiate this module directly.

Anchor:
- `src/diagnosis/evidence_retriever.py:1`

Recommendation (architectural):
- Keep a single authoritative evidence retrieval path for diagnosis to avoid divergence in retrieval semantics and scoring.

Persistence behavior for extended hallucination framework:
- Existing evaluator outputs remain unchanged:
	- `${results_dir}/{tag}_samples.jsonl`
	- `${results_dir}/{tag}_aggregate.json`
- New appended analysis outputs:
	- `${results_dir}/hallucination_samples.jsonl`
	- `${results_dir}/hallucination_samples.csv`

This preserves backward compatibility while enabling per-sample taxonomy inspection.

---

## 9) Practical Theoretical Risks in This RAG Design

1. Retrieval miss risk: faithful generator still fails if evidence is absent.
2. NLI threshold sensitivity: calibration drift changes SCR/CR balance.
3. Claim segmentation granularity: under/over-segmentation distorts metric denominator.
4. Multi-hop claims: CDEE proxy may not perfectly capture compositional failures.
5. Refusal pattern coverage: regex misses could over-penalize valid abstentions.

Relevant anchors:
- Retrieval config and top-k: `config/config.yaml:6`, `config/config.yaml:8`
- NLI thresholds: `config/config.yaml:37`, `config/config.yaml:38`
- Claim controls: `config/config.yaml:39`, `config/config.yaml:40`
- Metric logic: `src/diagnosis/metric_engine.py:72`

---

## 10) Minimal Reading Order for New Contributors

1. `README.md:28`
2. `src/pipeline.py:36`
3. `src/retrieval/retriever.py:15`
4. `src/generation/generator.py:44`
5. `src/diagnosis/diagnose.py:48`
6. `src/diagnosis/metric_engine.py:51`
7. `scripts/run_inference.py:91`
8. `tests/test_integration.py:64`

This order mirrors the runtime path from query to diagnosis and then to confidence in behavior through tests.
