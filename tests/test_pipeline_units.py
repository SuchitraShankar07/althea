"""
tests/test_pipeline_units.py
Unit tests for the core components.
Run with:  pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnosis.claim_extractor import SpacyClaimExtractor
from src.diagnosis.metric_engine import HallucinationMetrics, MetricEngine
from src.diagnosis.verification_engine import (
    ClaimVerificationResult,
    NLILabel,
    _is_temporal_claim,
)
from src.evaluation.evaluator import exact_match, f1_score


# ═══════════════════════════════════════════════════════
# Claim Extractor
# ═══════════════════════════════════════════════════════
class TestSpacyClaimExtractor:
    def test_fallback_split_basic(self):
        extractor = SpacyClaimExtractor.__new__(SpacyClaimExtractor)
        extractor.min_length = 4
        extractor.nlp = None
        text = "The Eiffel Tower is in Paris. It was built in 1889. Napoleon did not build it."
        claims = extractor.extract(text)
        assert len(claims) >= 2

    def test_empty_input(self):
        extractor = SpacyClaimExtractor.__new__(SpacyClaimExtractor)
        extractor.min_length = 4
        extractor.nlp = None
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []

    def test_max_claims_respected(self):
        extractor = SpacyClaimExtractor.__new__(SpacyClaimExtractor)
        extractor.min_length = 2
        extractor.nlp = None
        text = ". ".join([f"Claim number {i} is a fact" for i in range(50)]) + "."
        claims = extractor.extract(text, max_claims=10)
        assert len(claims) <= 10


# ═══════════════════════════════════════════════════════
# Temporal claim detection
# ═══════════════════════════════════════════════════════
class TestTemporalDetection:
    def test_temporal_keywords(self):
        assert _is_temporal_claim("The current president is X")
        assert _is_temporal_claim("As of 2023, the population is 8 billion")
        assert _is_temporal_claim("Recently, scientists discovered a new planet")
        assert _is_temporal_claim("The latest iPhone was released in 2024")

    def test_non_temporal(self):
        assert not _is_temporal_claim("The Eiffel Tower is located in Paris")
        assert not _is_temporal_claim("Water boils at 100 degrees Celsius")


# ═══════════════════════════════════════════════════════
# Metric Engine
# ═══════════════════════════════════════════════════════
class TestMetricEngine:
    def _make_result(self, label: NLILabel, is_temporal=False, is_outdated=False) -> ClaimVerificationResult:
        return ClaimVerificationResult(
            claim="test claim",
            label=label,
            entailment_score=1.0 if label == NLILabel.ENTAILMENT else 0.0,
            contradiction_score=1.0 if label == NLILabel.CONTRADICTION else 0.0,
            neutral_score=1.0 if label == NLILabel.NEUTRAL else 0.0,
            is_temporal=is_temporal,
            is_outdated=is_outdated,
        )

    def test_perfect_answers(self):
        engine = MetricEngine()
        results = [self._make_result(NLILabel.ENTAILMENT) for _ in range(5)]
        m = engine.compute(results)
        assert m.scr == 1.0
        assert m.cr == 0.0
        assert m.total_claims == 5
        assert m.supported_claims == 5

    def test_all_contradicted(self):
        engine = MetricEngine()
        results = [self._make_result(NLILabel.CONTRADICTION) for _ in range(4)]
        m = engine.compute(results)
        assert m.scr == 0.0
        assert m.cr == 1.0

    def test_mixed_results(self):
        engine = MetricEngine()
        results = [
            self._make_result(NLILabel.ENTAILMENT),
            self._make_result(NLILabel.ENTAILMENT),
            self._make_result(NLILabel.CONTRADICTION),
            self._make_result(NLILabel.NEUTRAL),
        ]
        m = engine.compute(results)
        assert m.scr == pytest.approx(0.5)
        assert m.cr == pytest.approx(0.25)
        assert m.total_claims == 4

    def test_temporal_claims(self):
        engine = MetricEngine()
        results = [
            self._make_result(NLILabel.NEUTRAL, is_temporal=True, is_outdated=True),
            self._make_result(NLILabel.ENTAILMENT, is_temporal=True, is_outdated=False),
            self._make_result(NLILabel.ENTAILMENT),
        ]
        m = engine.compute(results)
        assert m.temporal_claims == 2
        assert m.outdated_claims == 1
        assert m.tve == pytest.approx(0.5)

    def test_empty_results(self):
        engine = MetricEngine()
        m = engine.compute([])
        assert m.total_claims == 0
        assert m.chs == 0.0

    def test_composite_score_ordering(self):
        """Higher hallucination → higher CHS."""
        engine = MetricEngine()

        good = [self._make_result(NLILabel.ENTAILMENT) for _ in range(5)]
        bad = [self._make_result(NLILabel.CONTRADICTION) for _ in range(5)]

        m_good = engine.compute(good)
        m_bad = engine.compute(bad)
        assert m_good.chs < m_bad.chs

    def test_from_config(self):
        cfg = {
            "metrics": {
                "scr_weight": 1.0,
                "conflict_weight": 1.0,
                "tve_weight": 0.8,
                "cdee_weight": 0.8,
                "hallucination_lambda": 0.5,
            }
        }
        engine = MetricEngine.from_config(cfg)
        assert engine.scr_weight == 1.0
        assert engine.lambda_ == 0.5


# ═══════════════════════════════════════════════════════
# QA Metrics
# ═══════════════════════════════════════════════════════
class TestQAMetrics:
    def test_exact_match_identical(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_exact_match_case_insensitive(self):
        assert exact_match("paris", "Paris") == 1.0

    def test_exact_match_with_articles(self):
        assert exact_match("the Eiffel Tower", "Eiffel Tower") == 1.0

    def test_exact_match_different(self):
        assert exact_match("London", "Paris") == 0.0

    def test_f1_perfect(self):
        assert f1_score("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)

    def test_f1_partial_overlap(self):
        score = f1_score("quick brown", "the quick brown fox")
        assert 0.0 < score < 1.0

    def test_f1_no_overlap(self):
        assert f1_score("apple orange", "car truck") == 0.0

    def test_f1_empty(self):
        assert f1_score("", "something") == 0.0


# ═══════════════════════════════════════════════════════
# HallucinationMetrics dataclass
# ═══════════════════════════════════════════════════════
class TestHallucinationMetrics:
    def test_to_dict_roundtrip(self):
        m = HallucinationMetrics(scr=0.8, cr=0.1, tve=0.05, cdee=0.02, chs=0.12)
        d = m.to_dict()
        assert d["scr"] == 0.8
        assert d["cr"] == 0.1
        assert "chs" in d

    def test_summary_string(self):
        m = HallucinationMetrics(scr=0.9, cr=0.1, tve=0.0, cdee=0.0, chs=0.05,
                                  total_claims=10, supported_claims=9)
        s = m.summary()
        assert "SCR" in s
        assert "9/10" in s