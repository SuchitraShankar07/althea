# Althea Parameters and Fine-Tuning Deep Dive

## 1) Parameter Surface Overview

The project parameterization is split between:
- Runtime/master config in `config/config.yaml`
- CPU-safe profile in `config/config.cpu.yaml`
- Script-level CLI overrides in `scripts/train.py`, `scripts/run_inference.py`, `scripts/evaluate.py`

Primary sections:
- Retrieval: `config/config.yaml:6`
- Generation: `config/config.yaml:23`
- Diagnosis: `config/config.yaml:34`
- Metrics: `config/config.yaml:43`
- Training: `config/config.yaml:51`
- Evaluation: `config/config.yaml:73`
- Paths: `config/config.yaml:85`

CPU profile anchors:
- `config/config.cpu.yaml:1`
- `config/config.cpu.yaml:8`
- `config/config.cpu.yaml:31`

---

## 2) Retrieval Parameters

## 2.1 Core knobs

- `encoder_model`: embedding model identity.
- `top_k`: retrieved passages per query.
- `index_path`, `corpus_path`: storage pointers.
- `embedding_dim`: vector dimensionality.

Anchor examples:
- Top-k value: `config/config.yaml:8`
- Retriever consumption: `src/retrieval/retriever.py:48`
- Query retrieval behavior: `src/retrieval/retriever.py:35`

## 2.2 Tuning intuition

Let $k$ denote retrieved documents:
- Low $k$: lower latency, higher miss probability.
- High $k$: better recall but noisier context and higher token footprint.

Empirical principle for this stack:
- Increase $k$ only when NLI support ratio (SCR) rises faster than contradiction/noise growth (CR/CDEE).

Metric hooks:
- `src/diagnosis/metric_engine.py:72`

---

## 3) Generation Parameters

## 3.1 Model and decoding controls

Generation config includes:
- `model_name`
- `load_in_4bit`
- `device_map`
- `max_new_tokens`
- `temperature`
- `do_sample`
- `adapter_path`

Anchors:
- Model selection: `config/config.yaml:24`
- 4-bit switch: `config/config.yaml:25`
- Adapter path: `config/config.yaml:30`
- Generator config factory: `src/generation/generator.py:177`

## 3.2 Hardware-aware behavior

`RAGGenerator` enforces fail-fast for large-model CPU attempts.

Anchors:
- Size heuristic and guard: `src/generation/generator.py:60`
- Guard class context: `src/generation/generator.py:44`

Theoretical rationale:
- Fail-fast avoids hidden OOM/timeout states that corrupt experiment reproducibility.

## 3.3 Prompt-level controllability

Grounding policy is encoded via system instruction and prompt template.

Anchors:
- System prompt: `src/generation/generator.py:22`
- Prompt builder: `src/generation/generator.py:28`

Effect on tuning:
- Prompt policy and adapter optimization interact: a stronger grounding template can reduce hallucination pressure before any gradient update.

---

## 4) Diagnosis and Metric Parameters (as Training Signal Controls)

## 4.1 NLI calibration knobs

Diagnosis config drives label assignment boundaries:
- `entailment_threshold`
- `contradiction_threshold`
- `nli_batch_size`

Anchors:
- Config section: `config/config.yaml:34`
- NLI constructor and thresholds: `src/diagnosis/verification_engine.py:68`
- Config loader: `src/diagnosis/verification_engine.py:186`

Calibration consequence:
- Raising entailment threshold typically lowers SCR, potentially increasing CHS.
- Lowering contradiction threshold tends to raise CR.

## 4.2 Claim extraction knobs

- `claim_min_length`
- `claim_max_per_answer`

Anchors:
- Config values: `config/config.yaml:39`, `config/config.yaml:40`
- Extractor behavior: `src/diagnosis/claim_extractor.py:38`
- Diagnoser claim cap usage: `src/diagnosis/diagnose.py:151`

Bias-variance view:
- More claims increase sensitivity to local errors, but may amplify noisy sentence fragments.

## 4.3 CHS weighting knobs

Parameters:
- `hallucination_lambda`
- `scr_weight`
- `conflict_weight`
- `tve_weight`
- `cdee_weight`

Anchors:
- Config definitions: `config/config.yaml:44`
- Engine constructor: `src/diagnosis/metric_engine.py:58`
- Formula implementation: `src/diagnosis/metric_engine.py:118`

Optimization interpretation:
\[
\min_\theta \; \mathbb{E}[\text{CHS}(y_\theta, x)]
\]
subject to acceptable QA utility (F1/EM).

---

## 5) Fine-Tuning Stack: Classes and Strategy Modes

## 5.1 Training sample abstraction

Current trainer-side sample type:
- `TrainingSample` with `query`, `response`, `contexts`, `chs`, `retrieval_score`, plus dynamic attrs.

Anchors:
- Sample class: `src/training/qlora_trainer.py:18`
- Constructor signature: `src/training/qlora_trainer.py:20`

Important implementation nuance:
- `src/training/signal_generator.py` writes fields including `answer` and `prompt`, and trainer normalization attempts to reconcile schema differences.

Anchors:
- Signal conversion: `src/training/signal_generator.py:29`
- Trainer normalization path: `src/training/qlora_trainer.py:432`

## 5.2 QLoRA trainer internals

Core trainer class and DPO path:
- `QLoRATrainer`: `src/training/qlora_trainer.py:97`
- `train_dpo`: `src/training/qlora_trainer.py:136`
- DPO dataset conversion: `src/training/qlora_trainer.py:359`

Memory-oriented techniques in code:
- 4-bit quantization config
- gradient checkpointing
- tiny batch + accumulation
- fallback training paths when DPO fails

Anchors:
- Base-model loading helper: `src/training/qlora_trainer.py:53`
- Memory cleanup utility: `src/training/qlora_trainer.py:115`

## 5.3 Metric-guided trainer facade

Strategy multiplexer:
- Class: `src/training/qlora_trainer.py:404`
- Entry method: `src/training/qlora_trainer.py:417`
- DPO mode: `src/training/qlora_trainer.py:470`
- Rejection mode: `src/training/qlora_trainer.py:527`
- Metric-loss proxy mode: `src/training/qlora_trainer.py:564`

Conceptual mapping:

1. DPO mode:
- Convert each sample into preference tuples `(prompt, chosen, rejected)`.
- Quality proxy from CHS determines synthetic preference construction.

2. Rejection mode:
- Separate high-quality and low-quality responses by CHS bands.
- Pair good vs bad responses into preference pairs.

3. Metric-loss mode (implemented as weighted preference filtering):
- Prioritize higher-CHS-quality samples into DPO-like tuples.

## 5.4 Theory note on DPO objective

Classical DPO can be written as:
\[
\mathcal{L}_{DPO}(\theta) = -\log\sigma\left(\beta\left[\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x) - \log\pi_{ref}(y_w|x)+\log\pi_{ref}(y_l|x)\right]\right)
\]

In this repo, preference pairs are partially synthetic and CHS-conditioned, so alignment quality depends strongly on the fidelity of CHS as a proxy for human preference.

Beta source:
- `config/config.yaml:69`

---

## 6) Training Orchestration and Control Plane

## 6.1 Two-phase training script

`scripts/train.py` performs:
1. Data collection via pipeline inference+diagnosis.
2. Fine-tuning via `MetricGuidedQLoRATrainer`.

Anchors:
- Main function: `scripts/train.py:71`
- Query loading: `scripts/train.py:63`
- Trainer invocation: `scripts/train.py:165`

## 6.2 Config and CLI overrides

- Training method override occurs in script before trainer init.
- Output paths are ensured/created.

Anchor:
- Method override location: `scripts/train.py:89`

## 6.3 Makefile integration

- Main training target: `Makefile:113`
- Method presets:
  - DPO: `Makefile:124`
  - Rejection: `Makefile:127`
  - Metric loss: `Makefile:130`

---

## 7) CPU vs GPU Parameter Profiles

## 7.1 CPU profile intent

`config/config.cpu.yaml` lowers risk and resource load:
- disabled 4-bit mode
- null adapter by default
- conservative training defaults and target modules

Anchors:
- CPU model + quantization: `config/config.cpu.yaml:9`, `config/config.cpu.yaml:13`
- CPU adapter default: `config/config.cpu.yaml:14`
- LoRA module selection: `config/config.cpu.yaml:42`

## 7.2 GPU profile intent

Main config is oriented toward full workflow with adapter consumption:
- `adapter_path` pre-pointed to generated model dir
- stronger/expanded LoRA targets

Anchors:
- Main adapter path: `config/config.yaml:30`
- Main LoRA target modules: `config/config.yaml:64`

---

## 8) Parameter Interaction Matrix (Practical)

1. `top_k` × `max_new_tokens`:
- Larger retrieval context often demands larger output budget; otherwise model truncates reasoning.

2. NLI thresholds × CHS weights:
- Label calibration directly changes optimization signal magnitude.

3. `dpo_beta` × synthetic pair quality:
- High beta can overfit noisy pair generation.

4. `lora_r`, `lora_alpha`, dropout:
- Capacity-regularization balance; too small underfits, too large destabilizes low-data tuning.

Config anchors:
- `config/config.yaml:55` through `config/config.yaml:69`

---

## 9) Known Implementation Risks to Track

1. Schema drift between sample producers and trainer consumers.
- Signal generator writes `answer`/`prompt`; trainer canonical path favors `response` and has fallback extraction.
- Anchors: `src/training/signal_generator.py:39`, `src/training/qlora_trainer.py:432`

2. Synthetic rejected responses can be weak negatives.
- Anchor: `src/training/qlora_trainer.py:381`

3. Architecture-specific LoRA target assumptions may fail on unseen backbones.
- Anchor: `src/training/qlora_trainer.py:150`

4. Fallback training branches increase robustness but complicate strict experiment comparability.
- Anchor: `src/training/qlora_trainer.py:231`

---

## 10) Recommended Parameter Tuning Workflow

1. Lock retrieval first (`top_k`, corpus quality, index freshness).
2. Calibrate NLI thresholds on a small labeled dev set.
3. Freeze CHS weights during first finetune iteration.
4. Sweep training strategy (`dpo`, `rejection`, `metric_loss`) with fixed seeds.
5. Evaluate both utility and safety metrics jointly (F1/EM and CHS family).

Evaluation anchors used in this loop:
- `src/evaluation/evaluator.py:95`
- `src/evaluation/evaluator.py:133`
- `scripts/evaluate.py:84`
