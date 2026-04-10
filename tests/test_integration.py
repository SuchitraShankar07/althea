"""
tests/test_integration.py
End-to-end smoke test using mock components (no GPU / large model needed).

Run with:  pytest tests/test_integration.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnosis.claim_extractor import SpacyClaimExtractor
from src.diagnosis.diagnose import HallucinationDiagnoser
from src.diagnosis.metric_engine import MetricEngine
from src.diagnosis.verification_engine import (
    ClaimVerificationResult,
    NLILabel,
    NLIVerificationEngine,
)
from src.evaluation.evaluator import RAGEvaluator
from src.training.signal_generator import TrainingSignalGenerator


# ── Shared fixtures ───────────────────────────────────────────────────────────
DUMMY_DOCS = [
    {"doc_id": "d1", "text": "The Eiffel Tower is located in Paris, France.", "score": 0.9, "metadata": {}},
    {"doc_id": "d2", "text": "It was constructed between 1887 and 1889.", "score": 0.85, "metadata": {}},
]

DUMMY_ANSWER = (
    "The Eiffel Tower is located in Paris, France. "
    "It was built in 1889 by Gustave Eiffel. "
    "Currently it is the tallest structure in France."
)


def make_mock_retriever(docs=DUMMY_DOCS):
    retriever = MagicMock()
    retriever.retrieve.return_value = docs
    return retriever


def make_mock_verifier():
    """Returns a verifier that always says ENTAILMENT."""
    verifier = MagicMock(spec=NLIVerificationEngine)
    def fake_verify(claims, evidence_lists):
        return [
            ClaimVerificationResult(
                claim=c,
                label=NLILabel.ENTAILMENT,
                entailment_score=0.9,
                contradiction_score=0.05,
                neutral_score=0.05,
            )
            for c in claims
        ]
    verifier.verify_claims.side_effect = fake_verify
    return verifier


# ═══════════════════════════════════════════════════════
# Diagnosis pipeline integration
# ═══════════════════════════════════════════════════════
class TestDiagnosisPipeline:
    def _make_diagnoser(self):
        extractor = SpacyClaimExtractor.__new__(SpacyClaimExtractor)
        extractor.min_length = 4
        extractor.nlp = None

        verifier = make_mock_verifier()
        metric_engine = MetricEngine()
        retriever = make_mock_retriever()

        return HallucinationDiagnoser(
            retriever=retriever,
            claim_extractor=extractor,
            verifier=verifier,
            metric_engine=metric_engine,
            evidence_top_k=3,
            max_claims=20,
        )

    def test_diagnose_returns_output(self):
        diagnoser = self._make_diagnoser()
        output = diagnoser.diagnose(DUMMY_ANSWER, original_docs=DUMMY_DOCS)

        assert output.answer == DUMMY_ANSWER
        assert len(output.claims) > 0
        assert output.metrics.total_claims > 0

    def test_diagnose_perfect_entailment(self):
        """All ENTAILMENT → SCR=1.0, CR=0.0"""
        diagnoser = self._make_diagnoser()
        output = diagnoser.diagnose(DUMMY_ANSWER)

        assert output.metrics.scr == 1.0
        assert output.metrics.cr == 0.0

    def test_diagnose_empty_answer(self):
        diagnoser = self._make_diagnoser()
        output = diagnoser.diagnose("")
        assert output.metrics.total_claims == 0

    def test_diagnose_batch(self):
        diagnoser = self._make_diagnoser()
        answers = [DUMMY_ANSWER, "Paris is in France.", "Water boils at 100 degrees."]
        outputs = diagnoser.diagnose_batch(answers)
        assert len(outputs) == 3

    def test_refusal_answer_not_penalized(self):
        extractor = SpacyClaimExtractor.__new__(SpacyClaimExtractor)
        extractor.min_length = 4
        extractor.nlp = None

        verifier = MagicMock(spec=NLIVerificationEngine)

        diagnoser = HallucinationDiagnoser(
            retriever=make_mock_retriever(),
            claim_extractor=extractor,
            verifier=verifier,
            metric_engine=MetricEngine(),
        )

        refusal_answer = (
            "I cannot provide a factual answer because the context does not "
            "contain enough information about this question."
        )
        output = diagnoser.diagnose(refusal_answer)

        assert output.metrics.chs == 0.0
        assert output.metrics.scr == 1.0
        assert output.metrics.cr == 0.0
        assert output.metrics.cdee == 0.0
        assert output.metrics.total_claims == 1
        verifier.verify_claims.assert_not_called()

    def test_contradiction_diagnoser(self):
        """Verifier returns CONTRADICTION → CR=1.0"""
        extractor = SpacyClaimExtractor.__new__(SpacyClaimExtractor)
        extractor.min_length = 4
        extractor.nlp = None

        verifier = MagicMock(spec=NLIVerificationEngine)
        def contradiction_verify(claims, evidence_lists):
            return [
                ClaimVerificationResult(
                    claim=c, label=NLILabel.CONTRADICTION,
                    entailment_score=0.05, contradiction_score=0.9, neutral_score=0.05,
                )
                for c in claims
            ]
        verifier.verify_claims.side_effect = contradiction_verify

        diagnoser = HallucinationDiagnoser(
            retriever=make_mock_retriever(),
            claim_extractor=extractor,
            verifier=verifier,
            metric_engine=MetricEngine(),
        )
        output = diagnoser.diagnose(DUMMY_ANSWER)
        assert output.metrics.cr == 1.0
        assert output.metrics.scr == 0.0


# ═══════════════════════════════════════════════════════
# Evaluation integration
# ═══════════════════════════════════════════════════════
class TestEvaluator:
    def _make_metrics(self, scr=0.9, cr=0.1):
        from src.diagnosis.metric_engine import HallucinationMetrics
        return HallucinationMetrics(
            scr=scr, cr=cr, tve=0.0, cdee=0.0,
            chs=(1 - scr + cr) * 0.5,
            total_claims=10, supported_claims=int(scr * 10),
        )

    def test_evaluate_basic(self, tmp_path):
        evaluator = RAGEvaluator(results_dir=str(tmp_path))
        queries = ["What is the capital of France?", "Who built the Eiffel Tower?"]
        predictions = ["Paris", "Gustave Eiffel"]
        ground_truths = ["Paris", "Gustave Eiffel"]
        hm = [self._make_metrics(), self._make_metrics()]

        result = evaluator.evaluate(queries, predictions, ground_truths, hm, model_tag="test")
        assert result.avg_em == 1.0
        assert result.avg_f1 == 1.0
        assert result.n_samples == 2

    def test_compare_baseline_vs_tuned(self, tmp_path):
        from src.evaluation.evaluator import AggregateResult
        evaluator = RAGEvaluator(results_dir=str(tmp_path))

        baseline = AggregateResult(
            avg_f1=0.4, avg_em=0.3, avg_scr=0.6, avg_cr=0.3,
            avg_tve=0.1, avg_cdee=0.1, avg_chs=0.5, n_samples=50,
            model_tag="baseline",
        )
        tuned = AggregateResult(
            avg_f1=0.6, avg_em=0.5, avg_scr=0.8, avg_cr=0.1,
            avg_tve=0.05, avg_cdee=0.05, avg_chs=0.2, n_samples=50,
            model_tag="tuned",
        )
        delta = evaluator.compare(baseline, tuned)
        assert delta["Δ F1"] > 0        # F1 improved
        assert delta["Δ CHS"] < 0       # CHS decreased (better)
        assert delta["Δ SCR"] > 0       # SCR improved


# ═══════════════════════════════════════════════════════
# Training signal generator
# ═══════════════════════════════════════════════════════
class TestTrainingSignalGenerator:
    def test_convert_and_save(self, tmp_path):
        from src.diagnosis.diagnose import DiagnosisOutput
        from src.diagnosis.metric_engine import HallucinationMetrics

        sig_gen = TrainingSignalGenerator(
            output_path=str(tmp_path / "samples.jsonl")
        )
        queries = ["What is the capital of France?"]
        docs_list = [DUMMY_DOCS]
        metrics = HallucinationMetrics(scr=0.9, cr=0.1, chs=0.1)
        diag = DiagnosisOutput(
            answer="Paris is the capital.",
            claims=["Paris is the capital of France."],
            verification_results=[],
            metrics=metrics,
        )
        diag.answer = "Paris is the capital."

        samples = sig_gen.convert(queries, docs_list, [diag])
        assert len(samples) == 1
        assert samples[0].chs == metrics.chs
        assert samples[0].query == queries[0]

    def test_save_and_load_roundtrip(self, tmp_path):
        from src.training.qlora_trainer import TrainingSample

        path = str(tmp_path / "samples.jsonl")
        sig_gen = TrainingSignalGenerator(output_path=path)
        samples = [
            TrainingSample(query="Q1", answer="A1", chs=0.2, prompt="P1"),
            TrainingSample(query="Q2", answer="A2", chs=0.5, prompt="P2"),
        ]
        sig_gen.save(samples)
        loaded = TrainingSignalGenerator.load(path)
        assert len(loaded) == 2
        assert loaded[0].query == "Q1"
        assert loaded[1].chs == 0.5