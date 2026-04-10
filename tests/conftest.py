"""
tests/conftest.py
Session-scoped ML stub injection and fixtures for the full test suite.

All heavy dependencies (torch, transformers, peft, trl, faiss,
sentence_transformers, datasets, bitsandbytes, accelerate) are replaced
by lightweight stubs so tests run without a GPU or model downloads.

Run the suite with:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


# ── Session-level stub installation ──────────────────────────────────────────
def _install_stubs():
    """
    Install minimal module stubs for all heavy ML dependencies.
    Called once at import time — before any test collection happens.
    """

    # ── loguru ───────────────────────────────────────────────────────────────
    if "loguru" not in sys.modules:
        loguru = types.ModuleType("loguru")

        class _Logger:
            def info(self, *_, **__): pass
            def debug(self, *_, **__): pass
            def warning(self, *_, **__): pass
            def error(self, *_, **__): pass
            def trace(self, *_, **__): pass

        loguru.logger = _Logger()
        sys.modules["loguru"] = loguru

    # ── torch ────────────────────────────────────────────────────────────────
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.float16 = "float16"
        torch.float32 = "float32"
        torch.cuda = MagicMock()
        torch.cuda.is_available = lambda: False
        torch.no_grad = MagicMock(return_value=__import__("contextlib").nullcontext())
        torch.tensor = MagicMock(return_value=MagicMock())

        # torch.utils.data stub
        utils = types.ModuleType("torch.utils")
        utils.data = types.ModuleType("torch.utils.data")

        class _FakeDataset:
            """Minimal base class so MetricWeightedDataset can subclass it."""
            pass

        utils.data.Dataset = _FakeDataset
        torch.utils = utils
        sys.modules["torch"] = torch
        sys.modules["torch.utils"] = utils
        sys.modules["torch.utils.data"] = utils.data

    # ── transformers ─────────────────────────────────────────────────────────
    if "transformers" not in sys.modules:
        tf = types.ModuleType("transformers")
        tf.AutoModelForCausalLM = MagicMock()
        tf.AutoTokenizer = MagicMock()
        tf.BitsAndBytesConfig = MagicMock()
        tf.GenerationConfig = MagicMock()
        tf.DataCollatorForLanguageModeling = MagicMock()
        tf.Trainer = MagicMock()
        tf.TrainingArguments = MagicMock()
        tf.pipeline = MagicMock()
        sys.modules["transformers"] = tf

    # ── peft ─────────────────────────────────────────────────────────────────
    if "peft" not in sys.modules:
        peft = types.ModuleType("peft")
        peft.LoraConfig = MagicMock()
        peft.TaskType = MagicMock()
        peft.get_peft_model = MagicMock()
        peft.prepare_model_for_kbit_training = MagicMock()
        peft.PeftModel = MagicMock()
        sys.modules["peft"] = peft

    # ── trl ──────────────────────────────────────────────────────────────────
    if "trl" not in sys.modules:
        trl = types.ModuleType("trl")
        trl.DPOTrainer = MagicMock()
        trl.DPOConfig = MagicMock()
        trl.SFTTrainer = MagicMock()
        sys.modules["trl"] = trl

    # ── faiss ────────────────────────────────────────────────────────────────
    if "faiss" not in sys.modules:
        faiss = types.ModuleType("faiss")
        _mock_index = MagicMock()
        _mock_index.ntotal = 0
        faiss.IndexFlatIP = MagicMock(return_value=_mock_index)
        faiss.write_index = MagicMock()
        faiss.read_index = MagicMock(return_value=_mock_index)
        sys.modules["faiss"] = faiss

    # ── sentence_transformers ────────────────────────────────────────────────
    if "sentence_transformers" not in sys.modules:
        st = types.ModuleType("sentence_transformers")
        import numpy as np
        _mock_model = MagicMock()
        _mock_model.encode = MagicMock(
            return_value=np.random.rand(1, 384).astype("float32")
        )
        st.SentenceTransformer = MagicMock(return_value=_mock_model)
        sys.modules["sentence_transformers"] = st

    # ── datasets ─────────────────────────────────────────────────────────────
    if "datasets" not in sys.modules:
        ds = types.ModuleType("datasets")

        class _FakeDataset(dict):
            """Minimal Dataset-like object."""
            @classmethod
            def from_dict(cls, d):
                obj = cls(d)
                return obj

            @classmethod
            def from_list(cls, items):
                """Stub for Dataset.from_list used in DPO training."""
                obj = cls()
                obj._items = items
                return obj

            def map(self, fn, **kwargs):
                return self

            def __len__(self):
                if hasattr(self, '_items'):
                    return len(self._items)
                return 0

        ds.Dataset = _FakeDataset
        ds.load_dataset = MagicMock(return_value=iter([]))
        sys.modules["datasets"] = ds

    # ── bitsandbytes ─────────────────────────────────────────────────────────
    if "bitsandbytes" not in sys.modules:
        bb = types.ModuleType("bitsandbytes")
        sys.modules["bitsandbytes"] = bb

    # ── accelerate ───────────────────────────────────────────────────────────
    if "accelerate" not in sys.modules:
        acc = types.ModuleType("accelerate")
        sys.modules["accelerate"] = acc

    # ── spacy ────────────────────────────────────────────────────────────────
    if "spacy" not in sys.modules:
        spacy = types.ModuleType("spacy")
        spacy.load = MagicMock(side_effect=OSError("stub: spacy model not available"))
        sys.modules["spacy"] = spacy


# Install stubs immediately on module import (before any collection)
_install_stubs()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_documents():
    """Three realistic document dicts as returned by the retriever."""
    return [
        {
            "doc_id": "d1",
            "text": (
                "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
                "in Paris, France. It was constructed from 1887 to 1889 as the centerpiece "
                "of the 1889 World's Fair."
            ),
            "score": 0.92,
            "metadata": {"title": "Eiffel Tower"},
        },
        {
            "doc_id": "d2",
            "text": (
                "Gustave Eiffel's company designed and built the Eiffel Tower. "
                "Gustave Eiffel oversaw the construction personally."
            ),
            "score": 0.87,
            "metadata": {"title": "Gustave Eiffel"},
        },
        {
            "doc_id": "d3",
            "text": (
                "The tower stands 330 metres tall including its broadcast antenna. "
                "It was the tallest man-made structure in the world until 1930."
            ),
            "score": 0.81,
            "metadata": {"title": "Eiffel Tower"},
        },
    ]


@pytest.fixture
def sample_answer():
    """A factually correct answer about the Eiffel Tower."""
    return (
        "The Eiffel Tower is located in Paris, France. "
        "It was built by Gustave Eiffel's company between 1887 and 1889. "
        "The tower stands approximately 330 metres tall."
    )


@pytest.fixture
def hallucinated_answer():
    """An answer with a detectable hallucination (Napoleon claim)."""
    return (
        "The Eiffel Tower is located in Paris, France. "
        "Napoleon Bonaparte commissioned the tower in 1800. "
        "It was completed in 1889 by Gustave Eiffel."
    )


@pytest.fixture
def claim_extractor():
    """SpacyClaimExtractor configured with regex fallback (nlp=None)."""
    from src.diagnosis.claim_extractor import SpacyClaimExtractor

    extractor = SpacyClaimExtractor.__new__(SpacyClaimExtractor)
    extractor.min_length = 5
    extractor.nlp = None
    return extractor


@pytest.fixture
def metric_engine():
    """MetricEngine with default weights."""
    from src.diagnosis.metric_engine import MetricEngine

    return MetricEngine(
        scr_weight=1.0,
        cr_weight=1.0,
        tve_weight=0.8,
        cdee_weight=0.8,
        lambda_=0.5,
    )


@pytest.fixture
def mock_retriever(sample_documents):
    """MagicMock retriever that always returns sample_documents."""
    retriever = MagicMock()
    retriever.retrieve.return_value = sample_documents
    return retriever


@pytest.fixture
def entailment_verifier():
    """Mock NLI verifier that always returns ENTAILMENT (score 0.92)."""
    from src.diagnosis.verification_engine import ClaimVerificationResult, NLILabel

    verifier = MagicMock()

    def _always_entailment(claims, evidence_lists):
        return [
            ClaimVerificationResult(
                claim=c,
                label=NLILabel.ENTAILMENT,
                entailment_score=0.92,
                contradiction_score=0.04,
                neutral_score=0.04,
            )
            for c in claims
        ]

    verifier.verify_claims.side_effect = _always_entailment
    return verifier


@pytest.fixture
def contradiction_verifier():
    """Mock NLI verifier that always returns CONTRADICTION (score 0.91)."""
    from src.diagnosis.verification_engine import ClaimVerificationResult, NLILabel

    verifier = MagicMock()

    def _always_contradiction(claims, evidence_lists):
        return [
            ClaimVerificationResult(
                claim=c,
                label=NLILabel.CONTRADICTION,
                entailment_score=0.04,
                contradiction_score=0.91,
                neutral_score=0.05,
            )
            for c in claims
        ]

    verifier.verify_claims.side_effect = _always_contradiction
    return verifier


@pytest.fixture
def diagnoser(mock_retriever, claim_extractor, entailment_verifier, metric_engine):
    """
    Fully wired HallucinationDiagnoser using all mock components.
    Verifier defaults to ENTAILMENT.
    """
    from src.diagnosis.diagnose import HallucinationDiagnoser

    return HallucinationDiagnoser(
        retriever=mock_retriever,
        claim_extractor=claim_extractor,
        verifier=entailment_verifier,
        metric_engine=metric_engine,
        evidence_top_k=3,
        max_claims=15,
    )


@pytest.fixture
def training_samples():
    """Five TrainingSample objects with varied CHS scores."""
    from src.training.qlora_trainer import TrainingSample

    return [
        TrainingSample(
            query="What is the Eiffel Tower?",
            answer="The Eiffel Tower is in Paris, built in 1889.",
            chs=0.05,
            prompt="[P1]",
        ),
        TrainingSample(
            query="What is the Eiffel Tower?",
            answer="The Eiffel Tower was built by Napoleon in 1800.",
            chs=0.72,
            prompt="[P1]",
        ),
        TrainingSample(
            query="Who invented the telephone?",
            answer="Alexander Graham Bell invented the telephone in 1876.",
            chs=0.08,
            prompt="[P2]",
        ),
        TrainingSample(
            query="Who invented the telephone?",
            answer="The telephone was always around, invented by many people.",
            chs=0.55,
            prompt="[P2]",
        ),
        TrainingSample(
            query="What caused the 2008 crisis?",
            answer="The 2008 crisis was caused by subprime mortgage collapse.",
            chs=0.12,
            prompt="[P3]",
        ),
    ]
