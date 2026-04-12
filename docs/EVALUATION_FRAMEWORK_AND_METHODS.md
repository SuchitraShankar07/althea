# Evaluation Framework and Methods (Project Deep Dive)

## 1) Evaluation Philosophy in This Repository

The project evaluates two orthogonal properties:

1. Answer utility (task accuracy): F1 and EM.
2. Factual reliability under retrieval constraints: SCR/CR/TVE/CDEE/CHS.

Extended factual reliability also includes a taxonomy-level hallucination layer:
- Retrieval Conflict
- Overgeneralization
- Outdated Information
- Synthesis Error
- Overall Hallucination Score (OHS)

This means model selection is multi-objective rather than pure QA score optimization.

Primary anchors:
- QA + hallucination evaluator module: `src/evaluation/evaluator.py:1`
- Hallucination metric engine: `src/diagnosis/metric_engine.py:1`
- Baseline vs tuned script: `scripts/evaluate.py:1`

---

## 2) Formal Metric Definitions

## 2.1 QA utility metrics

### Exact Match (EM)

Answer strings are normalized before strict match:
- lowercasing
- article removal
- punctuation stripping
- whitespace compaction

Anchors:
- Normalizer: `src/evaluation/evaluator.py:24`
- EM function: `src/evaluation/evaluator.py:31`

### Token F1

F1 uses overlap between normalized token multisets:
\[
P = \frac{|\text{pred}\cap\text{gt}|}{|\text{pred}|},\quad
R = \frac{|\text{pred}\cap\text{gt}|}{|\text{gt}|},\quad
F1 = \frac{2PR}{P+R}
\]

Anchor:
- `src/evaluation/evaluator.py:35`

## 2.2 Hallucination metrics

Computed from claim verification outcomes:
- SCR (higher is better)
- CR (lower is better)
- TVE (lower is better)
- CDEE (lower is better)
- CHS (lower is better)

Extended taxonomy metrics (all normalized to [0, 1], higher means higher risk):
- retrieval_conflict
- overgeneralization
- outdated_information
- synthesis_error
- overall_hallucination_score (OHS)

Anchors:
- Metric dataclass: `src/diagnosis/metric_engine.py:23`
- Compute implementation: `src/diagnosis/metric_engine.py:72`

Composite score:
\[
CHS = \lambda\Big(w_{scr}(1-SCR)+w_{cr}CR+w_{tve}TVE+w_{cdee}CDEE\Big)
\]

Taxonomy aggregate score:

\[
OHS = 0.3\cdot RC + 0.25\cdot SE + 0.25\cdot OG + 0.2\cdot OI
\]

Where:
- $RC$ = retrieval_conflict
- $SE$ = synthesis_error
- $OG$ = overgeneralization
- $OI$ = outdated_information

Weight source:
- `config/config.yaml:44`

---

## 3) Claim-Level Verification Pipeline Behind Evaluation

To compute hallucination metrics, each generated answer is transformed through:

1. Claim extraction.
2. Per-claim evidence retrieval.
3. NLI classification.
4. Temporal and synthesis-aware metric aggregation.
5. Taxonomy-level hallucination detection and labeling.

Anchors:
- Diagnoser orchestration class: `src/diagnosis/diagnose.py:48`
- Main diagnose path: `src/diagnosis/diagnose.py:74`
- NLI engine: `src/diagnosis/verification_engine.py:68`
- Metric engine: `src/diagnosis/metric_engine.py:51`

This architecture makes evaluation interpretable: score regressions can be traced to claim extraction, evidence retrieval, label calibration, or aggregation policy.

Additional taxonomy detector anchors:
- Retrieval conflict detector: `src/diagnosis/metric_engine.py:123`
- Overgeneralization detector: `src/diagnosis/metric_engine.py:165`
- Outdated information detector: `src/diagnosis/metric_engine.py:195`
- Synthesis error detector: `src/diagnosis/metric_engine.py:230`

---

## 4) Evaluation Runner Workflow

## 4.1 Script process (`scripts/evaluate.py`)

1. Load config and dataset.
2. Evaluate baseline (`adapter_path = None`).
3. Optionally evaluate tuned adapter.
4. Compute and persist delta summary.

Anchors:
- Script main: `scripts/evaluate.py:100`
- Baseline run logic: `scripts/evaluate.py:120`
- Tuned run logic: `scripts/evaluate.py:130`
- Comparison save: `scripts/evaluate.py:161`

## 4.2 Batch inference and scoring loop

`evaluate_model` loops over all samples, runs full inference+diagnosis, then converts diagnosis dict into `HallucinationMetrics` objects for evaluator aggregation.

Anchor:
- `scripts/evaluate.py:84`

---

## 5) Dataset Layer and Benchmark Breadth

## 5.1 Evaluation-time datasets

Configured in evaluation script:
- HotpotQA
- Natural Questions
- PopQA

Anchors:
- Hotpot loader: `scripts/evaluate.py:37`
- NQ loader: `scripts/evaluate.py:49`
- PopQA loader: `scripts/evaluate.py:64`
- Loader registry: `scripts/evaluate.py:76`

## 5.2 Data preparation utility datasets

Broader preparation script also supports:
- TriviaQA
- RAGTruth
- Corpus preparation + merged train query export

Anchors:
- Dataset loaders registry: `scripts/prepare_data.py:118`
- TriviaQA loader: `scripts/prepare_data.py:79`
- RAGTruth loader: `scripts/prepare_data.py:96`
- Wikipedia corpus prep: `scripts/prepare_data.py:127`
- Training query extraction: `scripts/prepare_data.py:155`

Theoretical advantage:
- Mixing multi-hop, open-domain, and hallucination-focused corpora improves robustness checks across retrieval and generation failure modes.

---

## 6) Aggregation and Persistence Semantics

## 6.1 Per-sample outputs

Each sample output contains:
- query, prediction, ground truth
- F1, EM
- hallucination metric fields

Anchor:
- Save routine: `src/evaluation/evaluator.py:150`

Output path convention:
- `${results_dir}/{tag}_samples.jsonl`

Extended appended artifacts (non-overwriting):
- `${results_dir}/hallucination_samples.jsonl`
- `${results_dir}/hallucination_samples.csv`

These contain:
- per-sample taxonomy scores
- per-sample taxonomy labels
- confidence
- overall_hallucination_score
- timestamp and model_tag

## 6.2 Aggregate outputs

Aggregated result includes means for all utility and hallucination metrics and sample count.

Extended aggregate includes:
- avg_retrieval_conflict
- avg_overgeneralization
- avg_outdated_information
- avg_synthesis_error
- avg_overall_hallucination_score
- avg_confidence
- pct_hallucinated_outputs

Anchors:
- Aggregate dataclass: `src/evaluation/evaluator.py:63`
- Evaluation aggregate computation: `src/evaluation/evaluator.py:95`

Output path convention:
- `${results_dir}/{tag}_aggregate.json`

## 6.3 Baseline-vs-tuned delta

`compare` reports signed deltas for all major metrics.

Anchor:
- `src/evaluation/evaluator.py:133`

Interpretation rule:
- Positive delta is good for F1/EM/SCR.
- Negative delta is good for CR/TVE/CDEE/CHS.
- Negative delta is good for OHS.

## 6.4 Hallucination schema (per sample)

The extended diagnosis object includes this backward-compatible record:

```python
{
	"query": str,
	"response": str,
	"contexts": List[str],
	"hallucination_scores": {
		"retrieval_conflict": float,
		"overgeneralization": float,
		"outdated_information": float,
		"synthesis_error": float
	},
	"hallucination_labels": {
		"retrieval_conflict": bool,
		"overgeneralization": bool,
		"outdated_information": bool,
		"synthesis_error": bool
	},
	"confidence": float,
	"overall_hallucination_score": float
}
```

Integration anchor:
- `src/pipeline.py:76`

---

## 7) Visual Analytics Layer

`visualise_results.py` generates charts for metric communication and failure diagnosis:

- Radar plot: `scripts/visualise_results.py:59`
- CHS histogram: `scripts/visualise_results.py:100`
- Claim breakdown stacked bars: `scripts/visualise_results.py:127`
- CHS vs F1 scatter/trend: `scripts/visualise_results.py:156`
- Baseline vs tuned bars: `scripts/visualise_results.py:182`
- CHS-over-training curve: `scripts/visualise_results.py:212`

Why this matters theoretically:
- Distributional views reveal failure concentration, not just mean behavior.
- Correlation views (`CHS` vs `F1`) surface tradeoffs between utility and factuality.

---

## 8) Evaluation from Pipeline API (Alternative Path)

Beyond script usage, evaluator integration also exists in pipeline batch mode.

Anchor:
- `src/pipeline.py:88`

Behavior:
- If ground truths are provided and diagnosis is enabled, batch run pushes results to evaluator with a supplied model tag.

Non-breaking optional hooks:
- `enable_hallucination_eval=True` in pipeline inference and batch methods.
- CLI switches:
	- `scripts/run_inference.py --disable-hallucination-eval`
	- `scripts/evaluate.py --disable-hallucination-eval`
	- `scripts/train.py --disable-hallucination-eval`

This supports embedding evaluation inside custom experiments without invoking CLI script wrappers.

---

## 9) Testing Coverage for Evaluation Logic

Unit and integration tests include explicit checks for core metric behavior and compare semantics.

Notable anchors:
- Perfect-answer metric test: `tests/test_pipeline_units.py:80`
- CHS ordering sanity test: `tests/test_pipeline_units.py:127`
- Baseline-vs-tuned compare test: `tests/test_integration.py:193`
- Refusal non-penalty integration check: `tests/test_integration.py:114`

Theoretical value:
- These tests enforce directional correctness and edge-case policy consistency (especially abstention handling).

---

## 10) Common Evaluation Pitfalls and How This Repo Addresses Them

1. Metric myopia (single-score optimization):
- Mitigated via dual utility + hallucination reporting.
- Anchors: `src/evaluation/evaluator.py:95`, `src/diagnosis/metric_engine.py:72`

2. Dataset leakage and overfitting to one domain:
- Mitigated through multiple open-domain datasets and separate preparation flow.
- Anchors: `scripts/evaluate.py:76`, `scripts/prepare_data.py:118`

3. Penalizing honest abstention:
- Mitigated via explicit refusal detection with zero hallucination penalty.
- Anchors: `src/diagnosis/diagnose.py:26`, `src/diagnosis/diagnose.py:74`

4. Mean-only reporting:
- Mitigated via per-sample JSONL and visual diagnostics.
- Anchors: `src/evaluation/evaluator.py:150`, `scripts/visualise_results.py:236`

---

## 11) Recommended Evaluation Protocol for This Project

1. Run baseline with fixed dataset/split and max-samples.
2. Run tuned with identical evaluation settings.
3. Compare aggregate deltas.
4. Inspect per-sample outliers (largest CHS, largest F1 drops).
5. Review visual outputs for distribution shifts.
6. Re-check thresholds if SCR/CR changes look implausible.

Execution anchors:
- Make baseline: `Makefile:103`
- Make compare: `Makefile:106`
- Make visualise: `Makefile:140`

---

## 12) Quick Reference Map

- Evaluator core: `src/evaluation/evaluator.py:85`
- Hallucination metric engine: `src/diagnosis/metric_engine.py:51`
- NLI verification engine: `src/diagnosis/verification_engine.py:68`
- Evaluation runner script: `scripts/evaluate.py:100`
- Visualization runner: `scripts/visualise_results.py:236`

This map is the shortest path for auditing or extending the current evaluation methodology.
